# Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration (ECCV 2024)

[Shihao Zhou](https://joshyzhou.github.io/), [Jinshan Pan](https://jspan.github.io/), [Jinglei Shi](https://jingleishi.github.io/), [Duosheng Chen](https://github.com/Calvin11311), [Lishen Qu](https://github.com/qulishen) and [Jufeng Yang](https://cv.nankai.edu.cn/)

#### News
- **Jul 02, 2024:** FPro has been accepted to ECCV 2024 :tada: 
<hr />


## Training
### Derain
To train FPro on SPAD, you can run:
```sh
./train.sh Deraining/Options/Deraining_FPro_spad.yml
```
### Dehaze
To train FPro on SOTS, you can run:
```sh
./train.sh Dehaze/Options/RealDehazing_FPro.yml
```
### Deblur
To train FPro on GoPro, you can run:
```sh
./train.sh Motion_Deblurring/Options/Deblurring_FPro.yml
```
### Deraindrop
To train FPro on AGAN, you can run:
```sh
./train.sh Deraining/Options/RealDeraindrop_FPro.yml
```
### Demoire 
To train FPro on TIP18, you can run:
```sh
./train.sh Demoiring/Options/RealDemoiring_FPro.yml
```

## Evaluation
To evaluate FPro, you can refer commands in 'test.sh':
For evaluate on each dataset, you should uncomment corresponding line.


## Results
Experiments are performed for different image processing tasks including, rain streak removal, raindrop removal, haze removal, motion blur removal, moire pattern removal. 
Here is a summary table containing hyperlinks for easy navigation:
<table>
  <tr>
    <th align="left">Benchmark</th>
    <th align="center">Pretrained model</th>
    <th align="center">Visual Results</th>
  </tr>
  <tr>
    <td align="left">SPAD</td>
    <td align="center"><a href="https://pan.baidu.com/s/1lHWbvsFFpbvja_vEcvnpqA">(code:gd8j)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1duMbd4L0rvrWvxv9wnW2eg">(code:ntgp)</a></td>
  </tr>
  <tr>
    <td align="left">AGAN</td>
    <td align="center"><a href="https://pan.baidu.com/s/1Ki2kmibr515dCJmbdlpMhQ">(code:dqml)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1cPbbsNM6i5ufyzIqPJz60g">(code:ul55)</a></td>
  </tr>
  <tr>
    <td align="left">SOTS</td>
    <td align="center"><a href="https://pan.baidu.com/s/117lm0l06YW1RuFzDPLiMZA">(code:aagq)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1N-ZVnL3oGRy3voJ3Fl-YtQ">(code:9ssj)</a></td>
  </tr>
    <tr>
    <td align="left">GoPro</td>
    <td align="center"><a href="https://pan.baidu.com/s/1WjEISK2AntfdYOrrMwZOZw">(code:lhds)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1EkXTI968Cyu7UnKwdgymag">(code:764e)</a></td>
  </tr>
    <tr>
    <td align="left">TIP18</td>
    <td align="center"><a href="https://pan.baidu.com/s/1NPmeAIZkVz7DkLVJxuonIw">(code:l13v)</a></td>
    <td align="center"><a href="https://pan.baidu.com/s/1tLGRx2pvogS0Sl7fpmloNQ">(code:9und)</a></td>
  </tr>

</table>


## Citation
If you find this project useful, please consider citing:

    @inproceedings{zhou_ECCV2024_FPro,
      title={Seeing the Unseen: A Frequency Prompt Guided Transformer for Image Restoration},
      author={Zhou, Shihao and Pan, Jinshan and Shi, Jinglei and Chen, Duosheng and Qu, Lishen and Yang, Jufeng},
      booktitle={ECCV},
      year={2024}
    }

## Acknowledgement

This code borrows heavily from [Restormer](https://github.com/swz30/Restormer). 