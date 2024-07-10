## Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers

from einops import rearrange

##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)



##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, isAtt):
        super(TransformerBlock, self).__init__()
        self.isAtt = isAtt
        if self.isAtt:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.isAtt:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x

########### window operation#############
def window_partition(x, win_size, dilation_rate=1):
    B, H, W, C = x.shape
    if dilation_rate !=1:
        x = x.permute(0,3,1,2) # B, C, H, W
        assert type(dilation_rate) is int, 'dilation_rate should be a int'
        x = F.unfold(x, kernel_size=win_size,dilation=dilation_rate,padding=4*(dilation_rate-1),stride=win_size) # B, C*Wh*Ww, H/Wh*W/Ww
        windows = x.permute(0,2,1).contiguous().view(-1, C, win_size, win_size) # B' ,C ,Wh ,Ww
        windows = windows.permute(0,2,3,1).contiguous() # B' ,Wh ,Ww ,C
    else:
        x = x.view(B, H // win_size, win_size, W // win_size, win_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win_size, win_size, C) # B' ,Wh ,Ww ,C
    return windows

def window_reverse(windows, win_size, H, W, dilation_rate=1):
    # B' ,Wh ,Ww ,C
    B = int(windows.shape[0] / (H * W / win_size / win_size))
    x = windows.view(B, H // win_size, W // win_size, win_size, win_size, -1)
    if dilation_rate !=1:
        x = windows.permute(0,5,3,4,1,2).contiguous() # B, C*Wh*Ww, H/Wh*W/Ww
        x = F.fold(x, (H, W), kernel_size=win_size, dilation=dilation_rate, padding=4*(dilation_rate-1),stride=win_size)
    else:
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class lowFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak, num_heads,win_size=8, bias=False):
        super(lowFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ap_kv = nn.AdaptiveAvgPool2d(1)
        self.kv = nn.Conv2d(dim_bak, dim * 2, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d( dim, dim, kernel_size=1, bias=bias)

    def forward(self, feature, prompt_feature):
        b, c1,h,w = feature.shape
        _, c2,_,_ = prompt_feature.shape

        query = self.q(feature).reshape(b, h * w, self.num_heads, c1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        
        prompt_feature = self.ap_kv(prompt_feature)#.reshape(b, c2, -1).permute(0, 2, 1)
        key_value = self.kv(prompt_feature).reshape(b, 2*c1, -1).permute(0, 2, 1).contiguous().reshape(b, -1, 2, self.num_heads, c1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        key, value = key_value[0], key_value[1]

        attn = (query @ key.transpose(-2, -1).contiguous()) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ value)

        out = rearrange(out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)

        return out

class LinearProjection(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., bias=True, isQuery = True):
        super().__init__()
        self.isQuery =isQuery
        inner_dim = dim_head *  heads
        self.heads = heads
        if self.isQuery:
            self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        else:
            self.to_kv = nn.Linear(dim, 2*inner_dim, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        if self.isQuery:
            q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            q = q[0]
            return q
        else:
            C = self.inner_dim 
            kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4).contiguous()
            k, v = kv[0], kv[1] 
            return k,v

class highFrequencyPromptFusion(nn.Module):
    def __init__(self, dim, dim_bak,win_size, num_heads, qkv_bias=True, qk_scale=None, bias=False):
        super(highFrequencyPromptFusion, self).__init__()
        self.num_heads = num_heads
        self.win_size = win_size  # Wh, Ww
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.to_q = LinearProjection(dim,num_heads,dim//num_heads,bias=qkv_bias,isQuery=True)
        self.to_kv = LinearProjection(dim_bak,num_heads,dim//num_heads,bias=qkv_bias,isQuery=False)
        
        self.kv_dwconv = nn.Conv2d(dim_bak , dim_bak, kernel_size=3, stride=1, padding=1, groups=dim_bak, bias=bias)
        
        self.softmax = nn.Softmax(dim=-1)

        self.project_out = nn.Linear(dim, dim)

    def forward(self, query_feature, key_value_feature):

        b,c,h,w = query_feature.shape
        _,c_2,_,_ = key_value_feature.shape
        
        key_value_feature = self.kv_dwconv(key_value_feature)
        
        # partition windows
        query_feature = rearrange(query_feature, ' b c1 h w -> b h w c1 ', h=h, w=w)
        query_feature_windows = window_partition(query_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        query_feature_windows = query_feature_windows.view(-1, self.win_size * self.win_size, c)  # nW*B, win_size*win_size, C
        
        key_value_feature = rearrange(key_value_feature, ' b c2 h w -> b h w c2 ', h=h, w=w)
        key_value_feature_windows = window_partition(key_value_feature, self.win_size)  # nW*B, win_size, win_size, C  N*C->C
        key_value_feature_windows = key_value_feature_windows.view(-1, self.win_size * self.win_size, c_2)  # nW*B, win_size*win_size, C
        
        B_, N, C = query_feature_windows.shape
        
        query = self.to_q(query_feature_windows)
        query = query * self.scale
        
        key,value = self.to_kv(key_value_feature_windows)
        attn = (query @ key.transpose(-2, -1).contiguous())
        attn = attn.softmax(dim=-1)

        out = (attn @ value).transpose(1, 2).contiguous().reshape(B_, N, C)

        out = self.project_out(out)

        # merge windows
        attn_windows = out.view(-1, self.win_size, self.win_size, C)
        attn_windows = window_reverse(attn_windows, self.win_size, h, w)  # B H' W' C
        return rearrange(attn_windows, 'b h w c -> b c h w', h=h, w=w)

##########################################################################
## channel dynamic filters
class dynamic_filter_channel(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter_channel, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.conv_gate = nn.Conv2d(group*kernel_size**2, group*kernel_size**2, kernel_size=1, stride=1, bias=False)
        self.act_gate  = nn.Sigmoid()
        self.bn = nn.BatchNorm2d(group*kernel_size**2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')

        self.pad = nn.ReflectionPad2d(kernel_size//2)

        self.ap_1 = nn.AdaptiveAvgPool2d((1, 1))
        #self.ap_2 = nn.AdaptiveMaxPool2d((1, 1))

    def forward(self, x):
        identity_input = x 
        low_filter1 = self.ap_1(x)
        #low_filter2 = self.ap_2(x)
        low_filter = self.conv(low_filter1)
        low_filter = low_filter * self.act_gate(self.conv_gate(low_filter))
        low_filter = self.bn(low_filter)     

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c//self.group, self.kernel_size**2, h*w)

        n,c1,p,q = low_filter.shape
        low_filter = low_filter.reshape(n, c1//self.kernel_size**2, self.kernel_size**2, p*q).unsqueeze(2)
       
        low_filter = self.act(low_filter)
        # print('low_filter size',low_filter.shape)
        # print('low_filter n,c1,p,q',n,c1,p,q)
    
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        out_high = identity_input - low_part
        return low_part, out_high


class frequenctSpecificPromptGenetator(nn.Module):
    def __init__(self, dim=3,h=128,w=65, flag_highF=True):
        super().__init__()
        self.flag_highF = flag_highF
        k_size = 3
        if flag_highF:
            w = (w - 1) * 2
            self.w = w
            self.h = h
            self.weight = nn.Parameter(torch.randn(1,dim, h, w, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(dim, dim, (1,k_size), padding=(0, k_size//2), groups=dim),
                                      nn.Conv2d(dim, dim, (k_size,1), padding=(k_size//2, 0), groups=dim),
                                      nn.GELU())
        else:
            self.complex_weight = nn.Parameter(torch.randn(1,dim, h, w, 2, dtype=torch.float32) * 0.02)
            self.body = nn.Sequential(nn.Conv2d(2*dim,2*dim,kernel_size=1,stride=1),
                                    nn.GELU(),
                                    )
            

    def forward(self, ffm, H, W):
        if self.flag_highF:
            ffm = F.interpolate(ffm, size=(H, W), mode='bilinear')
            y_att = self.body(ffm)

            y_f = y_att * ffm
            y = y_f * self.weight

        else:
            ffm = F.interpolate(ffm, size=(H, W), mode='bicubic')
            y = torch.fft.rfft2(ffm.to(torch.float32).cuda())
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            weight = torch.complex(self.complex_weight[..., 0],self.complex_weight[..., 1])
            y_att = self.body(y_f)
            y_f = y_f * y_att
            y_real, y_imag = torch.chunk(y_f, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y = y * weight
            y = torch.fft.irfft2(y, s=(H, W))
        
        return y
    
##########################################################################
## PromptModule
class PromptModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()
        h = input_resolution
        w = input_resolution//2 +1
        self.simple_Fusion = nn.Conv2d(2*dim,dim,kernel_size=1,stride=1)

        self.FSPG_high = frequenctSpecificPromptGenetator(basic_dim,h,w, flag_highF=True)
        self.FSPG_low = frequenctSpecificPromptGenetator(basic_dim,h,w, flag_highF=False)


        self.modulator_hi = highFrequencyPromptFusion(dim, basic_dim, win_size=8, num_heads=2, bias=False)
        self.modulator_lo = lowFrequencyPromptFusion(dim, basic_dim, win_size=8, num_heads=2, bias=False)
    def forward(self, low_part, out_high , x):
        b,c,h,w = x.shape
        
        y_h = self.FSPG_high(out_high, h, w)
        y_l = self.FSPG_low(low_part, h, w)
        
        y_h = self.modulator_hi(x,y_h)
        y_l = self.modulator_lo(x,y_l)

        x = self.simple_Fusion(torch.cat([y_h,y_l], dim=1)) 

        return x

## PromptModule
class splitFrequencyModule(nn.Module):
    def __init__(self, basic_dim=32, dim=32, input_resolution=128):
        super().__init__()

        self.dyna_channel = dynamic_filter_channel(inchannels=basic_dim)
    def forward(self, F_low ):
        _,c_basic,h_ori, w_ori = F_low.shape

        low_part, out_high = self.dyna_channel(F_low)

        return low_part, out_high


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- FPro -----------------------
class FPro(nn.Module):
    def __init__(self, 
        inp_channels=3, 
        out_channels=3, 
        dim = 48,
        num_blocks = [4,6,6,8], 
        num_refinement_blocks = 4,
        heads = [1,2,4,8],
        ffn_expansion_factor = 2.66,
        bias = False,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
        dual_pixel_task = False        ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
    ):

        super(FPro, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[0])])
        
        self.down1_2 = Downsample(dim) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[1])])
        
        self.down2_3 = Downsample(int(dim*2**1)) ## From Level 2 to Level 3
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=False) for i in range(num_blocks[2])])

        self.splitFre =splitFrequencyModule(basic_dim= dim,dim=int(dim*2**2),input_resolution=32)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[2])])
        self.prompt_d3 = PromptModule(basic_dim= dim,dim=int(dim*2**2),input_resolution=64)

        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[1])])
        self.prompt_d2 = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=128)

        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_blocks[0])])
        self.prompt_d1 = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=256)

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, isAtt=True) for i in range(num_refinement_blocks)])
        self.prompt_r = PromptModule(basic_dim= dim,dim=int(dim*2**1),input_resolution=256)
        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim, int(dim*2**1), kernel_size=1, bias=bias)
        ###########################
            
        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img):

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)
        
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3) 


        out_dec_level3 = self.decoder_level3(out_enc_level3) 
        low_part, out_high = self.splitFre(inp_enc_level1)
        out_dec_level3 = self.prompt_d3(low_part, out_high,out_dec_level3) + out_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2) 
        out_dec_level2 = self.prompt_d2(low_part, out_high,out_dec_level2) + out_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        out_dec_level1 = self.prompt_d1(low_part, out_high,out_dec_level1) + out_dec_level1
        
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.prompt_r(low_part, out_high,out_dec_level1) + out_dec_level1

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img


        return out_dec_level1

