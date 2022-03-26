import os
import numpy as np

metrics = ['LPIPS', 'FID', 'PSNR', 'SSIM']
arrows = ['downarrow', 'downarrow', 'uparrow', 'uparrow']
datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
methods = ['SRGAN_official_x2_ema/visualization', 'SRGAN_ours_x2_ema/visualization',
           'ESRGAN_official_x2_ema/visualization', 'ESRGAN_ours_x2_ema/visualization', 'SwinIR_official_x2_ema/visualization', 'SwinIR_ours_x2_ema/visualization']
logs_root_dir = 'results/table_logs_all/all_avgs/'
table_txt = os.path.join(logs_root_dir, 'table_latex_texts_supp_x2_groupsort.txt')

with open(table_txt, 'a') as f:
    latex_txt = '*********\n' * 3
    f.write(latex_txt)

for index_metric in range(len(metrics)):

    metric = metrics[index_metric]
    for log_file in os.listdir(logs_root_dir):
        if metric.lower() in log_file.lower() and 'x2' in log_file:
            log_path = os.path.join(logs_root_dir, log_file)
    arrow = arrows[index_metric]
    with open(table_txt, 'a') as f:
        latex_txt = '\midrule\n\multirow{6}*{' + metric + ' $\{}$'.format(arrow) + '}\n'
        f.write(latex_txt)
    for dataset in datasets:
        with open(table_txt, 'a') as f:
            latex_txt = '&{}'.format(dataset)
            f.write(latex_txt)
        numbers = []
        for method in methods:
            log_path = log_path.replace('_swinir', '').replace('_swin', '')
            with open(log_path, 'r') as f_log:
                for line in f_log.readlines():
                    if method.lower() == line.lower().split('__')[0] or method.lower()+'_official' == line.lower().split('__')[0]:
                        if dataset.lower() in line.lower():
                            number = float(line.split(': ')[-1][:7])
                            numbers.append(number)
        print(numbers)

        for number_index in range(len(numbers)):
            if metric == 'PSNR' or metric == 'FID':
                latex_txt = '&{:.3f}'.format(numbers[number_index])
            else:
                latex_txt = '&{:.4f}'.format(numbers[number_index])
            with open(table_txt, 'a') as f:
                f.write(latex_txt)

        with open(table_txt, 'a') as f:
            latex_txt = '\_\_\n'.replace('_', '')
            f.write(latex_txt)
