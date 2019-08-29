from os import listdir

import gdal
import numpy as np
from gdalconst import GA_ReadOnly
from os.path import isfile, join
from PIL import Image

from keras_preprocessing.image import load_img

from area_assesment.geo.geotiff_utils import write_geotiff_cut_image

source_dir = '../../data/train/big_sat'
target_dir = '../../data/train/sat'
target_shape = (512, 512)
train_sat_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f)) and f.endswith('.tif')]

for i, f_sat in enumerate(train_sat_files):
    img_sat = np.array(load_img(join(source_dir, f_sat), grayscale=False))
    gdal_ds = gdal.Open(join(source_dir, f_sat), GA_ReadOnly)

    num_patches_row, num_patches_col = img_sat.shape[0] // target_shape[0], img_sat.shape[1] // target_shape[1]
    patches = np.array([img_sat[i: i + target_shape[0], j: j + target_shape[1]]
                         for i in range(0, num_patches_row * target_shape[0], target_shape[0])
                         for j in range(0, num_patches_col * target_shape[1], target_shape[1])])

    row = 0
    col = 0
    for image in patches:
        # temp_map = Image.fromarray(image)
        name_sat = join(target_dir, f_sat[:-4] + '_{}_{}.tif'.format(row, col))
        # temp_map.save(join(target_dir, name_sat))

        write_geotiff_cut_image(name_sat, gdal_ds, image, row, col, path_size=target_shape)

        col += 1
        if col > num_patches_col-1:
            col = 0
            row += 1

