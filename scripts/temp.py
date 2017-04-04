import cv2
import os
import numpy as np
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score

from area_assesment.geo.utils import write_geotiff
from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
from area_assesment.io_operations.visualization import plot2, plot3, plot_imgs
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.cnn_circle_farms import cnn_circle_farms_v1
from area_assesment.neural_networks.unet import *
import hashlib


sat_gen = ImageDataGenerator(
    rotation_range=90.)

mask_gen = ImageDataGenerator(
    rotation_range=90.)


output_folder = os.path.normpath('../sakaka_data/buildings/output/')

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
# dir = os.path.normpath('../sakaka_data/buildings/train2/sat/')
files_sat = filenames_in_dir(os.path.normpath('../sakaka_data/buildings/train2/sat/'), endswith_='.tif')[:4]
files_mask = filenames_in_dir(os.path.normpath('../sakaka_data/buildings/train2/map/'), endswith_='.tif')[:4]
sats = np.array([(cv2.imread(f).astype('float32')/255.0) for f in files_sat])
masks = np.array([(cv2.imread(f).astype('float32')/255.0) for f in files_mask])
print(sats.shape, masks.shape)
# sats = np.empty((0, sats[0], nn_input_patch_size[1], 3))
# masks = np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1], 2))
# for i, (f_sat, f_mask) in enumerate(list(zip(files_sat, files_mask))):
#     logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(files_sat), f_sat))
#     sat_, mask_ = cv2.imread(f_sat), cv2.imread(f_mask, cv2.IMREAD_GRAYSCALE)
#     plot2(sat_, mask_, show_plot=True)
#
#     sat, mask = sat_.astype('float32')/255.0, sat_.astype('float32')/255.0
#
#     sats, masks = sat[np.newaxis, :], mask[np.newaxis, :]
#     print('sats.shape: {}, masks.shape: {}'.format(sats.shape, masks.shape))


sat_gen.fit(sats)
mask_gen.fit(masks)

sat_flow = sat_gen.flow(sats, seed=0)
mask_flow = mask_gen.flow(masks, seed=0)

print(type(zip(sat_flow, mask_flow)))
for s, m in zip(sat_flow, mask_flow):
    print(s.shape, m.shape)
    sm = np.array([s, m])
    print('sm.shape', sm.shape)
    plot_imgs(sm)
    # plot_imgs(m)
