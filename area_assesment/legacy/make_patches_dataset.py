from os.path import join
import cv2
import os
from PIL import Image
import numpy as np
from os import listdir
from os.path import isfile, join
import logging
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches, rotateImage, array2patches_new
from area_assesment.io_operations.visualization import plot2
from area_assesment.neural_networks.cnn import *
from keras_preprocessing.image import load_img


# PATCHING SETTINGS
nn_input_patch_size = (64, 64)  # (1024, 1024)  # (1024, 1024)  # (64, 64)
nn_output_patch_size = (64, 64)  # (128, 128)  # (256, 256) # (16, 16)
step_size = 32  # 256  # 16

# COLLECT BIG IMAGES FROM DIRECTORY
dir_source = os.path.normpath('../../data/big_train/')
dir_target = os.path.normpath('../../data/train/')
dir_source_sat = os.path.join(dir_source, 'sat/')
dir_source_map = os.path.join(dir_source, 'map/')
dir_target_sat = os.path.join(dir_target, 'sat/')
dir_target_map = os.path.join(dir_target, 'map/')

logging.info('COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY: {}, {}'.format(dir_source_sat, dir_source_map))
train_sat_files = [f for f in listdir(dir_source_sat) if isfile(join(dir_source_sat, f)) and f.endswith('.tif')] # filenames_in_dir(dir_source_sat, endswith_='.tif')
train_map_files = [f for f in listdir(dir_source_map) if isfile(join(dir_source_map, f)) and f.endswith('.tif')] # filenames_in_dir(dir_source_map, endswith_='.tif')

sat_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3))
map_patches = np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1]))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    logging.info('LOADING IMG: {}/{}, {}, {}'.format(i + 1, len(train_map_files), f_sat, f_map))

    # img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)
    img_sat = np.array(load_img(join(dir_source_sat, f_sat), grayscale=False))
    img_map = np.array(load_img(join(dir_source_map, f_map), grayscale=True))

    # b, g, r = cv2.split(img_sat)  # get b,g,r
    # img_sat = cv2.merge([r, g, b])  # switch it to rgb

    # img_sat, img_map = img_sat, img_map
    logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    # print('Hash img_sat: ', hashlib.sha1(img_sat.view(np.uint8)).hexdigest())
    # print('Hash img_map: ', hashlib.sha1(img_map.view(np.uint8)).hexdigest())
    logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    img_sat = img_sat.astype('float32')
    ret, img_map = cv2.threshold(img_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    img_map = img_map.astype('float32')
    img_sat /= 255
    img_map /= 255

    # plot2(img_sat, img_map, show_plot=False, save_output_path='', name='img_and_map')

    # bgr_img = cv2.imread('../../data/Mecca.tif')
    # b, g, r = cv2.split(bgr_img)  # get b,g,r
    # rgb_img = cv2.merge([r, g, b])  # switch it to rgb

    # rgb_img = rgb_img.astype('float32')
    # rgb_img /= 255
    # q = array2patches_new(rgb_img[0:1023, 0:763], patch_size=nn_input_patch_size, step_size=64)
    # print(q.shape)
    #  / 0

    img_sat_patches = array2patches_new(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    img_map_patches = array2patches_new(img_map, patch_size=nn_input_patch_size, step_size=step_size)
    # img_sat_patches = extract_patches_2d(img_sat, nn_input_patch_size)
    # img_map_patches = extract_patches_2d(img_map, nn_input_patch_size)

    logging.info('sat_patches.shape: {}'.format(img_sat_patches.shape))
    logging.info('img_map_patches.shape: {}'.format(img_map_patches.shape))

    # Save masks and sats
    num_patches_col = (img_sat.shape[0] - nn_input_patch_size[0]) // step_size
    num_patches_row = (img_sat.shape[1] - nn_input_patch_size[1]) // step_size

    row = 0
    col = 0
    for j in range(len(img_sat_patches)):
        temp_sat = Image.fromarray((img_sat_patches[j] * 255).astype(np.uint8))
        temp_map = Image.fromarray((img_map_patches[j] * 255).astype(np.uint8))
        name_sat = train_sat_files[i][:-4] + '_{}_{}.tif'.format(row, col)
        name_map = train_map_files[i][:-4] + '_{}_{}.tif'.format(row, col)
        temp_sat.save(join(dir_target_sat, train_sat_files[i]))
        temp_map.save(join(dir_target_map, train_map_files[i]))
        col += 1
        if col > num_patches_col:
            col = 0
            num_patches_row += 1

    # sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)
