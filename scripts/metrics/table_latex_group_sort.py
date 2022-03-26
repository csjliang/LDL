import os
import numpy as np

metrics = ['LPIPS', 'DISTS', 'FID', 'PSNR', 'SSIM']
arrows = ['downarrow', 'downarrow', 'downarrow', 'uparrow', 'uparrow']
datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
methods = ['SRGAN_official_x2_ema/visualization', 'SRGAN_ours_x2_ema/visualization',
           'ESRGAN_official_x2_ema/visualization', 'ESRGAN_ours_x2_ema/visualization', 'SwinIR_official_x2_ema/visualization', 'SwinIR_ours_x2_ema/visualization']
logs_root_dir = 'results/table_logs_all/all_avgs/'
table_txt = os.path.join(logs_root_dir, 'table_latex_texts_supp_x2_groupsort.txt')

groups = [[0,2],[2,4],[4,6]]

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
            # if 'swinir' not in method.lower():
            log_path = log_path.replace('_swinir', '').replace('_swin', '')
            with open(log_path, 'r') as f_log:
                for line in f_log.readlines():
                    if method.lower() == line.lower().split('__')[0] or method.lower()+'_official' == line.lower().split('__')[0]:
                        if dataset.lower() in line.lower():
                            number = float(line.split(': ')[-1][:7])
                            numbers.append(number)
        print(numbers)
        highlight_pos = []
        for group_index in groups:
            left = group_index[0]
            right = group_index[1]
            print(numbers[left:right])
            if arrow == 'downarrow':
                highlight_pos.append(np.argsort(numbers[left:right])[0] + left)
            else:
                highlight_pos.append(np.argsort(numbers[left:right])[-1] + left)
        print(highlight_pos)

        for number_index in range(len(numbers)):
            if number_index not in highlight_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&{:.3f}'.format(numbers[number_index])
                else:
                    latex_txt = '&{:.4f}'.format(numbers[number_index])
            else:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_textbf{'.replace('_', '') + '{:.3f}'.format(numbers[number_index]) + '}'
                else:
                    latex_txt = '&\_textbf{'.replace('_', '') + '{:.4f}'.format(numbers[number_index]) + '}'
            with open(table_txt, 'a') as f:
                f.write(latex_txt)

        with open(table_txt, 'a') as f:
            latex_txt = '\_\_\n'.replace('_', '')
            f.write(latex_txt)
