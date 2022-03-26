import os
import numpy as np

logs_root_dir = 'results/table_logs_all/all_avgs/'

metrics = ['LPIPS', 'DISTS', 'FID', 'PSNR', 'SSIM']
arrows = ['downarrow', 'downarrow', 'downarrow', 'uparrow', 'uparrow']
datasets = ['Set5', 'Set14', 'Manga109', 'General100', 'Urban100', 'DIV2K100']
methods = ['SRGAN', 'SFTGAN', 'ESRGAN', 'USRGAN', 'SPSR', 'SRGAN_ours_DIV2K', 'ESRGAN_ours_DIV2K', 'ESRGAN_ours_DF2K']
table_txt = os.path.join(logs_root_dir, 'table_latex_texts.txt')

with open(table_txt, 'a') as f:
    latex_txt = '*********\n' * 3
    f.write(latex_txt)

for index_metric in range(len(metrics)):

    metric = metrics[index_metric]
    for log_file in os.listdir(logs_root_dir):
        if metric.lower() in log_file.lower() and 'all_avgs' in log_file:
            log_path = os.path.join(logs_root_dir, log_file)
    arrow = arrows[index_metric]
    with open(table_txt, 'a') as f:
        latex_txt = '\midrule\n\multirow{7}*{' + metric + ' $\{}$'.format(arrow) + '}\n'
        f.write(latex_txt)
    for dataset in datasets:
        with open(table_txt, 'a') as f:
            latex_txt = '&{}'.format(dataset)
            f.write(latex_txt)
        numbers_sota = []
        numbers_ours = []
        for method in methods:
            with open(log_path, 'r') as f_log:
                for line in f_log.readlines():
                    if method.lower() == line.lower().split('__')[0] or method.lower()+'_official' == line.lower().split('__')[0]:
                        if dataset.lower() in line.lower():
                            number = float(line.split(': ')[-1][:7])
                            if 'ours' not in line:
                                numbers_sota.append(number)
                            else:
                                numbers_ours.append(number)
        if arrow == 'downarrow':
            highlight_sota_pos = np.argmin(numbers_sota)
            highlight_ours_pos = np.argmin(numbers_ours)
        else:
            highlight_sota_pos = np.argmax(numbers_sota)
            highlight_ours_pos = np.argmax(numbers_ours)
        for number_index in range(len(numbers_sota)):
            if number_index != highlight_sota_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&{:.3f}'.format(numbers_sota[number_index])
                else:
                    latex_txt = '&{:.4f}'.format(numbers_sota[number_index])
            else:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_textcolor{blue}{'.replace('_', '') + '{:.3f}'.format(numbers_sota[number_index]) + '}'
                else:
                    latex_txt = '&\_textcolor{blue}{'.replace('_', '') + '{:.4f}'.format(numbers_sota[number_index]) + '}'
            with open(table_txt, 'a') as f:
                f.write(latex_txt)
        for number_index in range(len(numbers_ours)):
            if number_index != highlight_ours_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&{:.3f}'.format(numbers_ours[number_index])
                else:
                    latex_txt = '&{:.4f}'.format(numbers_ours[number_index])
            else:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_textbf{\_textcolor{red}{'.replace('_', '') + '{:.3f}'.format(numbers_ours[number_index]) + '}}'
                else:
                    latex_txt = '&\_textbf{\_textcolor{red}{'.replace('_', '') + '{:.4f}'.format(numbers_ours[number_index]) + '}}'
            with open(table_txt, 'a') as f:
                f.write(latex_txt)
        with open(table_txt, 'a') as f:
            latex_txt = '\_\_\n'.replace('_', '')
            f.write(latex_txt)
