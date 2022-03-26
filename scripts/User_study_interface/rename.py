import os
import shutil

root_dir = 'all_images'
target_dir = 'all_images_'
os.makedirs(target_dir, exist_ok=True)

for file in os.listdir(root_dir):
    source_path = os.path.join(root_dir, file)
    if 'GT' not in file:
        target_name = file.replace(file.split('_')[0] + '__', '')
        target_path = os.path.join(target_dir, target_name)
    else:
        target_path = os.path.join(target_dir, file)

    shutil.copy(source_path, target_path)

# root_dir = 'all_images'
# target_dir = 'all_images_jpg'
# os.makedirs(target_dir, exist_ok=True)
#
# for file in os.listdir(root_dir):
#     source_path = os.path.join(root_dir, file)
#     img = cv2.imread(source_path)
#     if 'GT' not in file:
#         target_name = file.replace(file.split('_')[0] + '__', '').split('.png')[0]+'.jpg'
#         target_path = os.path.join(target_dir, target_name)
#     else:
#         target_path = os.path.join(target_dir, file.split('.png')[0]+'.jpg')
#     cv2.imwrite(target_path, img)
