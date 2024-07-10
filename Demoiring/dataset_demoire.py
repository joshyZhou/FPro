import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_png_file, load_img, Augment_RGB_torch, loader4demoire
import torch.nn.functional as F
import random
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
from torchvision import transforms
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'thin_target'
        input_dir = 'thin_source'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]

        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target

        # self.transform = transforms.Compose([
        #                       transforms.Scale(532),
        #                       transforms.RandomCrop(512),
        #                       # transforms.RandomHorizontalFlip(),
        #                       # transforms.ToTensor(),
        #                       # transforms.Normalize(mean, std),
        #                     ])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        ps = self.img_options['patch_size']
        # for Tip18
        clean = torch.from_numpy(np.float32(loader4demoire(self.clean_filenames[tar_index],ps)))
        noisy = torch.from_numpy(np.float32(loader4demoire(self.noisy_filenames[tar_index],ps)))
        # clean = loader4demoire(self.clean_filenames[tar_index],ps)
        # noisy = loader4demoire(self.noisy_filenames[tar_index],ps)
        # clean,noisy = self.transform(clean,noisy)
        #
        # clean = clean.resize((ps, ps), Image.ANTIALIAS)
        # clean = np.float32(clean)
        # clean = clean / 255.
        # clean = torch.from_numpy(clean)
        #
        # noisy = noisy.resize((ps, ps), Image.ANTIALIAS)
        # noisy = np.float32(noisy)
        # noisy = noisy / 255.
        # noisy = torch.from_numpy(noisy)

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target

        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        #cancel the aug strategy
        # apply_trans = transforms_aug[random.getrandbits(3)]
        #
        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'thin_target'
        input_dir = 'thin_source'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

        # self.transform = transforms.Compose([
        #     transforms.Scale(532),
        #     transforms.CenterCrop(512),
        #     # transforms.RandomCrop(img_options['patch_size']),
        #     # transforms.RandomHorizontalFlip(),
        #     # transforms.ToTensor(),
        #     # transforms.Normalize(mean, std),
        # ])

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        # # for Tip18
        clean = torch.from_numpy(np.float32(loader4demoire(self.clean_filenames[tar_index],286)))
        noisy = torch.from_numpy(np.float32(loader4demoire(self.noisy_filenames[tar_index],286)))
        # clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        # noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        # clean = loader4demoire(self.clean_filenames[tar_index], 512)
        # noisy = loader4demoire(self.noisy_filenames[tar_index], 512)
        # clean, noisy = self.transform(clean, noisy)
        #
        # clean = clean.resize((256, 256), Image.ANTIALIAS)
        # clean = np.float32(clean)
        # clean = clean / 255.
        # clean = torch.from_numpy(clean)
        #
        # noisy = noisy.resize((256, 256), Image.ANTIALIAS)
        # noisy = np.float32(noisy)
        # noisy = noisy / 255.
        # noisy = torch.from_numpy(noisy)

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = 256
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        # if H - ps == 0:
        #     r = 0
        #     c = 0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        r = (H-ps)//2
        c = (H-ps)//2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        # apply_trans = transforms_aug[random.getrandbits(3)]
        #
        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderTest, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'thin_target'
        input_dir = 'thin_source'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))

        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]

        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):

        tar_index = index % self.tar_size

        # for Tip18
        clean = torch.from_numpy(np.float32(loader4demoire(self.clean_filenames[tar_index],256)))
        noisy = torch.from_numpy(np.float32(loader4demoire(self.noisy_filenames[tar_index],256)))

        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = 256
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        # if H - ps == 0:
        #     r = 0
        #     c = 0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        r = (H - ps) // 2
        c = (H - ps) // 2
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]


        # apply_trans = transforms_aug[random.getrandbits(3)]
        #
        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)