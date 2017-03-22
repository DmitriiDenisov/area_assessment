import matplotlib.pyplot as plt
import numpy as np
import cv2
from area_assesment.io_operations.data_io import filenames_in_dir

dir_train = '../../sakaka_data/train/'  # '../../data/mass_buildings/train/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tif')[:3]
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')[:3]
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    print('PATCHING IMG: {}/{}, {}, {}'.format(i + 1, len(train_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    ret, img_map = cv2.threshold(img_map, 127, 255, cv2.THRESH_BINARY)
    print(img_map.max())
    plt.imshow(img_map)
    plt.show()