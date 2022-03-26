
import cv2
import os

patch_indexes = [[123, 440], [529, 996], [817, 873], [1220, 1683], [196, 513], [459, 94], [788, 524], [980, 872], [1292, 1196], [443, 345]]
image_indexes = ['2_', '4_', '5_', '6_', '10_', '12_', '13_', '26_', '28_', '31_']
img_dir = 'results/Compare_RealESRGAN'
target_dir = 'results/Compare_RealESRGAN_selected_patches'
os.makedirs(target_dir, exist_ok=True)

patch_sizes = [100, 100, 100, 120, 86, 64, 90, 60, 108, 100]
ratio = 1.5

for i in range(len(image_indexes)):

    image_index = image_indexes[i]
    patch_index = patch_indexes[i]
    # patch_index_HR = patch_indexes_HR[i]
    for file in os.listdir(img_dir):
        if image_index == file[:len(image_index)]:
            img_path = os.path.join(img_dir, file)
            target_path = os.path.join(target_dir, file)
            # if not os.path.exists(target_path):
            if True:
                print(img_path)
                img = cv2.imread(img_path)
                patch = img[patch_index[1]:patch_index[1] + patch_sizes[i], patch_index[0]:patch_index[0] + int(patch_sizes[i] * ratio), :]
                cv2.imwrite(target_path, patch)
            # if 'HR' in file:
            #     patch = img[patch_index_HR[1]:patch_index_HR[1] + patch_sizes_HR[i], patch_index_HR[0]:patch_index_HR[0] + patch_sizes_HR[i], :]
            #     target_path = os.path.join(target_dir, file.split('.')[0] + '_largeHR.png')
            #     cv2.imwrite(target_path, patch)
