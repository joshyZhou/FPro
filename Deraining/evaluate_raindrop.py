import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def calc_psnr(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_psnr(im1_y, im2_y)


def calc_ssim(im1, im2):
    im1_y = cv2.cvtColor(im1, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    im2_y = cv2.cvtColor(im2, cv2.COLOR_BGR2YCR_CB)[:, :, 0]
    return compare_ssim(im1_y, im2_y)


def align_to_four(img):
    a_row = int(img.shape[0]/4)*4
    a_col = int(img.shape[1]/4)*4
    img = img[0:a_row, 0:a_col, :]
    return img


def evaluate_raindrop(in_dir, gt_dir):
    inputs = sorted(glob(os.path.join(in_dir, '*.png')) + glob(os.path.join(in_dir, '*.jpg')))
    gts = sorted(glob(os.path.join(gt_dir, '*.png')) + glob(os.path.join(gt_dir, '*.jpg')))
    psnrs = []
    ssims = []
    for input, gt in tqdm(zip(inputs, gts)):
        inputdata = cv2.imread(input)
        gtdata = cv2.imread(gt)
        inputdata = align_to_four(inputdata)
        gtdata = align_to_four(gtdata)
        psnrs.append(calc_psnr(inputdata, gtdata))
        ssims.append(calc_ssim(inputdata, gtdata))

    ave_psnr = np.array(psnrs).mean()
    ave_ssim = np.array(ssims).mean()
    return ave_psnr, ave_ssim


if __name__ == '__main__':
    ave_psnr, ave_ssim = evaluate_raindrop('/mnt/sda/zsh/FPro/Deraining/results/FPro_AGAN/test_a', '/mnt/sda/zsh/dataset/test_a/gt')
    print('PSNR: ', ave_psnr)
    print('SSIM: ', ave_ssim)
