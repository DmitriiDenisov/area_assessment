import cv2
import os
import numpy as np
# import logging
# from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
# from sklearn.feature_extraction.image import *
# from sklearn.metrics import jaccard_similarity_score

# from area_assesment.geo.utils import write_geotiff
# from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
# from area_assesment.images_processing.patching import array2patches
from area_assesment.io_operations.visualization import plot2, plot3, plot_imgs
# from area_assesment.neural_networks.cnn import *
# from area_assesment.neural_networks.cnn_circle_farms import cnn_circle_farms_v1
from area_assesment.neural_networks.unet import *
# import hashlib


data_gen_args = dict(rotation_range=90.,
                     rescale=1/255.0,
                     fill_mode='reflect',
                     horizontal_flip=True,
                     vertical_flip=True)

sat_gen = ImageDataGenerator(**data_gen_args)
mask_gen = ImageDataGenerator(**data_gen_args)

output_folder = os.path.normpath('.')

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
files_sat = filenames_in_dir(os.path.normpath('../../data/old_train/sat/'), endswith_='.tif')[10:11]
files_mask = filenames_in_dir(os.path.normpath('../../data/old_train/map'), endswith_='.tif')[10:11]
sats = np.array([(cv2.imread(f).astype('float32')) for f in files_sat])
masks = np.array([(cv2.imread(f).astype('float32')) for f in files_mask])
print(sats.shape, masks.shape)

sat_gen.fit(sats)
mask_gen.fit(masks)

sat_flow = sat_gen.flow(sats)
mask_flow = mask_gen.flow(masks)

print(type(zip(sat_flow, mask_flow)))
i = 0
for s, m in zip(sat_flow, mask_flow):
    print(s.shape, m.shape)
    sm = np.array([s, m])
    print('sm.shape', sm.shape)
    plot_imgs(sm, name='TEST_{}'.format(i), save_output_path='')
    i += 1
