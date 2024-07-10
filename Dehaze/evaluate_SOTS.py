## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
import argparse
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
import utils

def proc(filename):
    tar,prd = filename
    prd_name = prd.split('/')[-1].split('_')[0]+'.png'
    tar_name = '/mnt/sda/zsh/dataset/haze/promptIR/outdoor/gt/' + prd_name
    # print('tar',tar)
    # print('prd',prd)
    tar_img = utils.load_img(tar_name)
    prd_img = utils.load_img(prd)
        
    PSNR = utils.calculate_psnr(tar_img, prd_img)
    SSIM = utils.calculate_ssim(tar_img, prd_img)
    return PSNR,SSIM

parser = argparse.ArgumentParser(description='Dehazing using FPro')

args = parser.parse_args()


datasets = ['outdoor']

for dataset in datasets:

    gt_path = os.path.join('/mnt/sda/zsh/dataset/haze/promptIR/outdoor/gt')
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.tif')))
    assert len(gt_list) != 0, "Target files not found"

    
    file_path = os.path.join('results', 'FPro', dataset)
    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.tif')))
    assert len(path_list) != 0, "Predicted files not found"

    psnr, ssim = [], []
    img_files =[(i, j) for i,j in zip(gt_list,path_list)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])

    avg_psnr = sum(psnr)/len(psnr)
    avg_ssim = sum(ssim)/len(ssim)

    # print('For {:s} dataset PSNR: {:f}\n'.format(dataset, avg_psnr))
    print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
