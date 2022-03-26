import matplotlib.pyplot as plt
import numpy as np
import os


metrics = ['LPIPS', 'DISTS', 'FID', 'PSNR', 'SSIM']
arrows = ['downarrow', 'downarrow', 'downarrow', 'uparrow', 'uparrow']
datasets = ['Set5', 'Set14', 'Manga109', 'General100', 'Urban100', 'DIV2K100']
methods = ['SRGAN', 'SFTGAN', 'ESRGAN', 'USRGAN', 'SPSR', 'SRGAN_ours_DIV2K', 'ESRGAN_ours_DIV2K_ema', 'ESRGAN_ours_DF2K_ema', 'SwinIR_official_ema', 'SwinIR_ours_ema']
methods_legend = ['SRGAN', 'SFTGAN', 'ESRGAN', 'USRGAN', 'SPSR', 'SRResNet_LDL', 'RRDB_LDL(DIV2K)', 'RRDB_LDL(DF2K)', 'SwinIR+ESRGAN', 'SwinIR+LDL']
marks = ['bo', 'bo', 'bx', 'bx', 'bx', 'ro', 'rx', 'rx', 'bs', 'rs']

logs_root_dir = 'results/table_logs_all/all_avgs/'
table_txt_file = os.path.join(logs_root_dir, 'table_latex_texts_final.txt')
figs_save_dir = 'results/plotted_figs_final'
os.makedirs(figs_save_dir, exist_ok=True)

values_all = np.zeros([len(datasets)*len(metrics), len(methods)])

with open(table_txt_file, 'r') as f:
    i = 0
    lines = f.readlines()
    for line in lines:
        if '&' in line:
            txt = line[1:-3].replace(line[1:-3].split('&')[0], '').replace('\\textcolor{blue}{', '').replace('}', '').replace('\\textcolor{red{', '').replace('\\textbf{', '')[1:]
            print(txt)
            values = txt.split('&')

            for j, value in enumerate(values):
                values_all[i, j] = float(value)
            i += 1

metric_1 = 'PSNR'
metric_2 = 'LPIPS'
dataset = 'DIV2K100'
for metric_1 in ['LPIPS', 'DISTS', 'FID']:
    for metric_2 in ['PSNR', 'SSIM']:
        for dataset in datasets:

            index_metric_1 = metrics.index(metric_1) * len(datasets) + datasets.index(dataset)
            index_metric_2 = metrics.index(metric_2) * len(datasets) + datasets.index(dataset)

            p1_ = values_all[index_metric_1, :]
            p2 = values_all[index_metric_2, :]

            p1 = []
            for p in p1_:
                p1.append(p)

            plt.figure('Draw')

            for i in range(len(p1)):
                plt.plot(p1[i], p2[i], marks[i], label=methods_legend[i])
            # plt.plot(p1[:-5], p2[:-5], 'rx', label = 'a')
            # plt.plot(p1[-5:], p2[-5:], 'bo', label = 'b')

            plt.xlabel('{}'.format(metric_1))
            plt.ylabel('{}'.format(metric_2))

            plt.title('{} vs {} on {}'.format(metric_1, metric_2, dataset))

            for i in range(len(p1)):
                plt.text(p1[i]*1.01, p2[i], r'{}'.format(methods_legend[i]), fontsize=10)

            plt.grid(True)
            # plt.gca().invert_xaxis()
            # plt.legend(loc="lower right", ncol=2)
            plt.savefig(os.path.join(figs_save_dir, metric_1 + '_' + metric_2 + '_' + dataset + '.png'))

            plt.close()