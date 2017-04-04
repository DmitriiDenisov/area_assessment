import cv2
import os
import numpy as np
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score

from area_assesment.geo.geotiff_utils import write_geotiff
from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
from area_assesment.io_operations.visualization import plot_img_mask, plot_img_mask_pred
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.cnn_circle_farms import cnn_circle_farms_v1
from area_assesment.neural_networks.unet import *
import hashlib


output_folder = os.path.normpath('../sakaka_data/buildings/output/')

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir = os.path.normpath('../sakaka_data/buildings/output/test_temp/')
files = filenames_in_dir(dir, endswith_='.tif')
for i, f in enumerate(files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(files), f))
    img_ = cv2.imread(f)  # , cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)
    # plot_img_mask(img_sat_, img_map_, show_plot=True)

    img = img_.astype('float32')
    img2 = img.copy()
    img2[img2 < np.floor(0.1 * 255)] = 0
    img2[img2 >= np.floor(0.1 * 255)] = 1
    plot_img_mask(img_, img2, show_plot=True)
    # geotiff_output_path = os.path.join(output_folder, '8-3.tif')
    # write_geotiff(geotiff_output_path, raster_layers=(img * 255).astype('int'), gdal_ds=f)
