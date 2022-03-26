import argparse
import cv2
import os
import warnings
import glob

from basicsr.metrics import calculate_niqe

log_save_path = 'results/table_logs_all/'

data_root = 'results/Compare'
ref_root = 'datasets/'
ref_dirs = ['DIV2K/DIV2K_valid_HR/']
datasets = ['DIV2K100']
methods = ['LDL']

logoverall_path = log_save_path + 'all_avgs/NIQE_all_avgs.txt'

for index in range(len(ref_dirs)):
    ref_dir = os.path.join(ref_root, ref_dirs[index])
    for method in methods:
        img_dir = os.path.join(data_root, method, datasets[index])
        print(img_dir)

        img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

        os.makedirs(log_save_path, exist_ok=True)
        log_path = log_save_path + 'NIQE__' + method + '__' + datasets[index] + '.txt'

        if not os.path.exists(log_path):

            crop_border = 4

            niqe_all = []

            for i, img_path in enumerate(img_list):
                file_name = img_path.split('/')[-1]
                basename, ext = os.path.splitext(os.path.basename(img_path))
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    niqe_score = calculate_niqe(img, crop_border, input_order='HWC', convert_to='y')
                niqe_all.append(niqe_score.item())
                log = f'{i + 1:3d}: {file_name:25}. \tNIQE: {niqe_score.item():.6f}.'
                with open(log_path, 'a') as f:
                    f.write(log + '\n')

            log = f'Average: NIQE_all: {sum(niqe_all) / len(niqe_all):.6f}'
            with open(log_path, 'a') as f:
                f.write(log + '\n')
            log_overall = method + '__' + datasets[index] + '__' + log
            with open(logoverall_path, 'a') as f:
                f.write(log_overall + '\n')
            print(log_overall)