import cv2
import os
import numpy as np
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score

from area_assesment.data_processing.DataGeneratorCustom import DataGeneratorCustom
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches, rotateImage, array2patches_new
from area_assesment.io_operations.visualization import plot2
from area_assesment.neural_networks.cnn import *
import hashlib

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 images_augmentation.py
#########################################################
from area_assesment.neural_networks.logger import TensorBoardBatchLogger
from area_assesment.neural_networks.unet import unet

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])


# MODEL DEFINITION
logging.info('MODEL DEFINITION')

if True:
    # model = unet(64, 64, 3)
    # model.summary()
    pass
else:
    # LOADING PREVIOUS WEIGHTS OF MODEL
    logging.info('LOADING PREVIOUS WEIGHTS OF MODEL: {}'.format(net_weights_load))
    model.load_weights(net_weights_load)



# PATCHING SETTINGS
nn_input_patch_size = (64, 64) # (1024, 1024)  # (1024, 1024)  # (64, 64)
nn_output_patch_size = (64, 64) # (128, 128)  # (256, 256) # (16, 16)
step_size = 64  # 256  # 16

# MODEL SETTINGS
epochs = 100
net_weights_load = None  # os.path.normpath('../weights/cnn_circlefarms/w_epoch97_jaccard0.6741_valjaccard0.4200.hdf5')
net_weights_dir_save = os.path.normpath('../../weights/unet_mecca')
########################################################

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir_train = os.path.normpath('../../data/old_train/')  # '../../data/mass_buildings/train/' # '/storage/_pdata/sakaka/circle_farms/train/'
dir_train_sat = os.path.join(dir_train, 'sat/')
dir_train_map = os.path.join(dir_train, 'map/')
logging.info('COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY: {}, {}'.format(dir_train_sat, dir_train_map))
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tif')[10:11]
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')[10:11]

sat_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3))
map_patches = np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1]))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    logging.info('LOADING IMG: {}/{}, {}, {}'.format(i + 1, len(train_map_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)

    b, g, r = cv2.split(img_sat)  # get b,g,r
    img_sat = cv2.merge([r, g, b])  # switch it to rgb

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

    i = 0
    for (sat_patch, map_patch) in list(zip(img_sat_patches, img_map_patches)):
        # logging.debug(sat_patch.shape, map_patch.shape)
        # plot2(sat_patch, map_patch, show_plot=False, save_output_path='', name='sat_map_{}'.format(i))
        i += 1

    img_map_patches = img_map_patches[:,
                                      nn_input_patch_size[0]//2 - nn_output_patch_size[0]//2:
                                      nn_input_patch_size[0]//2 + nn_output_patch_size[0]//2,
                                      nn_input_patch_size[1]//2 - nn_output_patch_size[1]//2:
                                      nn_input_patch_size[1]//2 + nn_output_patch_size[1]//2]
    logging.debug('sat_patches.shape: {}'.format(img_sat_patches.shape))
    logging.debug('img_map_patches.shape: {}'.format(img_map_patches.shape))
    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)
print('sat_patches.shape: {}'.format(sat_patches.shape))
print('map_patches.shape: {}'.format(map_patches.shape))

del img_sat
del img_map

print('augmenting images...')
angles = [90, 180, 270]
init_shape_sat = sat_patches.shape
init_shape_map = map_patches.shape
for k in range(len(angles)):
    print('rotating by {} degrees'.format(angles[k]))
    sat_patches_rotate = np.zeros(init_shape_sat, np.float32)
    map_patches_rotate = np.zeros(init_shape_map, np.float32)
    for i in range(init_shape_sat[0]):
        for j in range(init_shape_sat[-1]):
            sat_patches_rotate[i, :, :, j] = rotateImage(sat_patches[i, :, :, j], angles[k])
        map_patches_rotate[i, :, :] = rotateImage(map_patches[i, :, :], angles[k])
    sat_patches, map_patches = np.append(sat_patches, sat_patches_rotate, 0), \
                               np.append(map_patches, map_patches_rotate, 0)
    for u in [(k+1)*64+1, (k+1)*64+36, 64+45*(k+1), 64*(k+1)+48, 64*(k+1)+54, 64*(k+1)+56]:
        plot2(sat_patches[u], map_patches[u], show_plot=False, save_output_path='', name='rot_{}_angle{}'.format(u-64, angles[k]))

print('augmented sat_patches.shape: {}'.format(sat_patches.shape))
print('augmented map_patches.shape: {}'.format(map_patches.shape))
1/0

# FIT MODEL AND SAVE WEIGHTS
logging.info('FIT MODEL, EPOCHS: {}, SAVE WEIGHTS: {}'.format(epochs, net_weights_dir_save))

tb_callback = TensorBoardBatchLogger(project_path='../../', batch_size=1, log_every=1)
checkpoint = ModelCheckpoint(os.path.join(net_weights_dir_save,
                             'w_epoch{epoch:02d}_jaccard{jaccard_coef:.4f}_valjaccard{val_jaccard_coef:.4f}.hdf5'),
                             monitor='val_jaccard_coef', save_best_only=True)
model.fit(sat_patches, map_patches, epochs=epochs, callbacks=[checkpoint, tb_callback], batch_size=32, validation_split=0.1)
