import os
import numpy as np

logs_root_dir = 'results/table_logs_all/all_avgs/'

metrics = ['LPIPS', 'PieAPP', 'DISTS', 'FID', 'PSNR', 'SSIM']
arrows = ['downarrow', 'downarrow', 'downarrow', 'downarrow', 'uparrow', 'uparrow']
datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
methods = ['SRGAN', 'SRGAN_ours_DIV2K', 'SFTGAN', 'ESRGAN', 'USRGAN', 'SPSR', 'SPSR_DF2K', 'ESRGAN_ours_DIV2K', 'ESRGAN_ours_DF2K']
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
        numbers = []
        for method in methods:
            with open(log_path, 'r') as f_log:
                for line in f_log.readlines():
                    if method.lower() == line.lower().split('__')[0] or method.lower()+'_official' == line.lower().split('__')[0]:
                        if dataset.lower() in line.lower():
                            number = float(line.split(': ')[-1][:7])
                            numbers.append(number)
        if arrow == 'downarrow':
            highlight_pos = np.argsort(numbers)[0]
            second_highlight_pos = np.argsort(numbers)[1]
            third_highlight_pos = np.argsort(numbers)[2]
        else:
            highlight_pos = np.argsort(numbers)[-1]
            second_highlight_pos = np.argsort(numbers)[-2]
            third_highlight_pos = np.argsort(numbers)[-3]

        for number_index in range(len(numbers)):
            if number_index != highlight_pos and number_index != second_highlight_pos and number_index != third_highlight_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&{:.3f}'.format(numbers[number_index])
                else:
                    latex_txt = '&{:.4f}'.format(numbers[number_index])
            elif number_index == second_highlight_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_textcolor{blue}{'.replace('_', '') + '{:.3f}'.format(numbers[number_index]) + '}'
                else:
                    latex_txt = '&\_textcolor{blue}{'.replace('_', '') + '{:.4f}'.format(numbers[number_index]) + '}'
            elif number_index == highlight_pos:
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_textbf{\_textcolor{red}{'.replace('_', '') + '{:.3f}'.format(numbers[number_index]) + '}}'
                else:
                    latex_txt = '&\_textbf{\_textcolor{red}{'.replace('_', '') + '{:.4f}'.format(numbers[number_index]) + '}}'
            elif number_index == third_highlight_pos:
                print(11)
                if metric == 'PSNR' or metric == 'FID':
                    latex_txt = '&\_underline{'.replace('_', '') + '{:.3f}'.format(numbers[number_index]) + '}'
                else:
                    latex_txt = '&\_underline{'.replace('_', '') + '{:.4f}'.format(numbers[number_index]) + '}'
            with open(table_txt, 'a') as f:
                f.write(latex_txt)

        with open(table_txt, 'a') as f:
            latex_txt = '\_\_\n'.replace('_', '')
            f.write(latex_txt)
