import argparse
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import os

from basicsr.data import build_dataset
from basicsr.metrics.fid import calculate_fid, extract_inception_features, load_patched_inception_v3

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

def calculate_fid_folder():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_sample', type=int, default=1000)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--backend', type=str, default='disk', help='io backend for dataset. Option: disk, lmdb')
    args = parser.parse_args()

    data_root = 'results/Compare'
    datasets = ['DIV2K100']
    methods = ['LDL']
    FID_stats = ['inception_Set5_512.pth', 'inception_Set14_512.pth', 'inception_Manga109_512.pth',
                 'inception_General100_512.pth', 'inception_Urban100_512.pth', 'inception_DIV2K100_512.pth']

    FID_stats_dir = 'results/FID_stats'
    log_save_path = 'results/table_logs_all/'
    logoverall_path = log_save_path + 'all_avgs/FID_all_avgs.txt'

    for index in range(len(datasets)):
        for method in methods:
            img_dir = os.path.join(data_root, method, datasets[index])
            fid_stats = os.path.join(FID_stats_dir, FID_stats[index])
            print(img_dir)

            os.makedirs(log_save_path, exist_ok=True)
            log_path = log_save_path + 'FID__' + method + '__' + datasets[index] + '.txt'

            if not os.path.exists(log_path):

                # inception model
                inception = load_patched_inception_v3(device)

                # create dataset
                opt = {}
                opt['name'] = 'SingleImageDataset'
                opt['type'] = 'SingleImageDataset'
                opt['dataroot_lq'] = img_dir
                opt['io_backend'] = dict(type=args.backend)
                opt['mean'] = [0.5, 0.5, 0.5]
                opt['std'] = [0.5, 0.5, 0.5]
                dataset = build_dataset(opt)

                # create dataloader
                data_loader = DataLoader(
                    dataset=dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    sampler=None,
                    drop_last=False)
                args.num_sample = min(args.num_sample, len(dataset))
                total_batch = math.ceil(args.num_sample / args.batch_size)

                def data_generator(data_loader, total_batch):
                    for idx, data in enumerate(data_loader):
                        if idx >= total_batch:
                            break
                        else:
                            yield data['lq']

                features = extract_inception_features(data_generator(data_loader, total_batch), inception, total_batch, device)
                features = features.numpy()
                total_len = features.shape[0]
                features = features[:args.num_sample]
                print(f'Extracted {total_len} features, ' f'use the first {features.shape[0]} features to calculate stats.')

                sample_mean = np.mean(features, 0)
                sample_cov = np.cov(features, rowvar=False)

                # load the dataset stats
                stats = torch.load(fid_stats)
                real_mean = stats['mean']
                real_cov = stats['cov']
                # calculate FID metric
                fid = calculate_fid(sample_mean, sample_cov, real_mean, real_cov)
                log = f'Average: FID: {fid:.6f}'
                with open(log_path, 'a') as f:
                    f.write(log + '\n')
                log_overall = method + '__' + datasets[index] + '__' + log
                with open(logoverall_path, 'a') as f:
                    f.write(log_overall + '\n')
                print(log_overall)
                assert 0 == 1


if __name__ == '__main__':
    calculate_fid_folder()
