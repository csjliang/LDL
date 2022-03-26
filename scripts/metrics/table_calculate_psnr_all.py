import cv2
import numpy as np
from os import path as osp
import os
import glob

from basicsr.metrics import calculate_psnr, calculate_ssim
from basicsr.utils import scandir
from basicsr.utils.matlab_functions import bgr2ycbcr


def main():
    """Calculate PSNR and SSIM for images.

    Configurations:
        folder_gt (str): Path to gt (Ground-Truth).
        folder_restored (str): Path to restored images.
        crop_border (int): Crop border for each side.
        suffix (str): Suffix for restored images.
        test_y_channel (bool): If True, test Y channel (In MatLab YCbCr format)
            If False, test RGB channels.
    """

    log_save_path = 'results/table_logs_all/'

    data_root = 'results/Compare'
    ref_root = 'datasets/'
    ref_dirs = ['DIV2K/DIV2K_valid_HR/']
    datasets = ['DIV2K100']
    methods = ['LDL']

    logoverall_path = log_save_path + 'all_avgs/PSNR_all_avgs.txt'

    for index in range(len(ref_dirs)):
        ref_dir = os.path.join(ref_root, ref_dirs[index])
        for method in methods:
            img_dir = os.path.join(data_root, method, datasets[index])
            print(img_dir)

            img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

            os.makedirs(log_save_path, exist_ok=True)
            log_path = log_save_path + 'PSNR__' + method + '__' + datasets[index] + '.txt'

            if not os.path.exists(log_path):

                crop_border = 4
                suffix = ''
                test_y_channel = True
                correct_mean_var = False
                # -------------------------------------------------------------------------

                psnr_all = []

                if test_y_channel:
                    print('Testing Y channel.')
                else:
                    print('Testing RGB channels.')

                for i, img_path in enumerate(img_list):
                    basename, ext = osp.splitext(osp.basename(img_path))
                    try:
                        img_restored = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(
                            np.float32) / 255.

                        file_name = img_path.split('/')[-1]
                        if 'DIV2K100' in img_dir and 'SFTGAN' not in img_dir:
                            gt_path = os.path.join(ref_dir, file_name[:4] + '.png')
                        elif 'Urban100' in img_dir and 'SFTGAN' not in img_dir:
                            gt_path = os.path.join(ref_dir, file_name[:7] + '.png')
                        elif 'SFTGAN' in img_dir:
                            ref_dir_SFTGAN = '/data1/liangjie/BasicSR_ALL/results/Compare/SFTGAN_official/GT'
                            gt_path = os.path.join(ref_dir_SFTGAN, file_name.split('_')[0] + '_gt.png')
                            if 'Urban100' in img_dir:
                                gt_path = os.path.join(ref_dir_SFTGAN,
                                                       file_name.split('_')[0] + '_' + file_name.split('_')[
                                                           1] + '_gt.png')
                        else:
                            if '_' in file_name:
                                gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '.png')
                            else:
                                gt_path = os.path.join(ref_dir, file_name)

                        img_gt = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

                        if correct_mean_var:
                            mean_l = []
                            std_l = []
                            for j in range(3):
                                mean_l.append(np.mean(img_gt[:, :, j]))
                                std_l.append(np.std(img_gt[:, :, j]))
                            for j in range(3):
                                # correct twice
                                mean = np.mean(img_restored[:, :, j])
                                img_restored[:, :,
                                             j] = img_restored[:, :, j] - mean + mean_l[j]
                                std = np.std(img_restored[:, :, j])
                                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                                mean = np.mean(img_restored[:, :, j])
                                img_restored[:, :,
                                             j] = img_restored[:, :, j] - mean + mean_l[j]
                                std = np.std(img_restored[:, :, j])
                                img_restored[:, :, j] = img_restored[:, :, j] / std * std_l[j]

                        if test_y_channel and img_gt.ndim == 3 and img_gt.shape[2] == 3:
                            img_gt = bgr2ycbcr(img_gt, y_only=True)
                            img_restored = bgr2ycbcr(img_restored, y_only=True)

                        # calculate PSNR
                        psnr = calculate_psnr(
                            img_gt * 255,
                            img_restored * 255,
                            crop_border=crop_border,
                            input_order='HWC')
                        psnr_all.append(psnr)
                        log = f'{i + 1:3d}: {file_name:25}. \tPSNR: {psnr:.6f}.'
                        with open(log_path, 'a') as f:
                            f.write(log + '\n')
                    except:
                        pass
                log = f'Average: PSNR: {sum(psnr_all) / len(psnr_all):.6f}'
                with open(log_path, 'a') as f:
                    f.write(log + '\n')
                log_overall = method + '__' + datasets[index] + '__' + log
                with open(logoverall_path, 'a') as f:
                    f.write(log_overall + '\n')
                print(log_overall)


if __name__ == '__main__':
    main()
