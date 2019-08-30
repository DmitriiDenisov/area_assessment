import cv2
import os
import numpy as np
import logging
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score

from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
# from area_assesment.io_operations.visualization import plot_img_mask
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.unet import unet
from area_assesment.neural_networks.uresnet import uresnet
import hashlib

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 model_train.py
#########################################################


logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])

# MODEL DEFINITION
logging.info('MODEL DEFINITION')
model = uresnet(64, 64, 3)
model.summary()

# PATCHING SETTINGS
nn_input_patch_size = (64, 64)  # (1024, 1024)  # (64, 64)
nn_output_patch_size = (64, 64)  # (256, 256) # (16, 16)
# step_size = 16  # 256  # 16

# MODEL SETTINGS
epochs = 100
net_weights_load = None#os.path.normpath('../weights/unet/w_epoch12_jaccard0.0819_valjaccard0.2943.hdf5')
net_weights_dir_save = os.path.normpath('../weights/unet/')
########################################################

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir_train = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/buildings/train/')  # '../../data/mass_buildings/train/'
dir_train_sat = os.path.join(dir_train, 'sat/')
dir_train_map = os.path.join(dir_train, 'map/')
logging.info('COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY: {}, {}'.format(dir_train_sat, dir_train_map))
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tif')[:3]
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')[:3]

sat_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3))
map_patches = np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1]))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    logging.info('LOADING IMG: {}/{}, {}, {}'.format(i + 1, len(train_map_files), f_sat, f_map))
    img_sat_, img_map = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)
    img_sat = equalizeHist_rgb(img_sat_)
    # img_sat, img_map = img_sat[-1000:, -1000:], img_map[-1000:, -1000:]
    logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    # print('Hash img_sat: ', hashlib.sha1(img_sat.view(np.uint8)).hexdigest())
    # print('Hash img_map: ', hashlib.sha1(img_map.view(np.uint8)).hexdigest())
    logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    img_sat = img_sat.astype('float32')
    ret, img_map = cv2.threshold(img_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    img_map = img_map.astype('float32')

    img_sat = (img_sat - img_sat.mean())/img_sat.std()  # img_sat /= 255
    img_map /= 255
    # plot_img_mask(img_sat, img_map, show_plot=True)

    # img_sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    # img_map_patches = array2patches(img_map, patch_size=nn_input_patch_size, step_size=step_size)
    img_sat_patches = extract_patches_2d(img_sat, nn_input_patch_size, max_patches=10000, random_state=1)
    img_map_patches = extract_patches_2d(img_map, nn_input_patch_size, max_patches=10000, random_state=1)
    # for (sat_patch, map_patch) in list(zip(img_sat_patches, img_map_patches)):
    #     logging.debug(sat_patch.shape, map_patch.shape)
    #     plot_img_mask(sat_patch, map_patch, show_plot=True)
    img_map_patches = img_map_patches[:,
                                      nn_input_patch_size[0]//2 - nn_output_patch_size[0]//2:
                                      nn_input_patch_size[0]//2 + nn_output_patch_size[0]//2,
                                      nn_input_patch_size[1]//2 - nn_output_patch_size[1]//2:
                                      nn_input_patch_size[1]//2 + nn_output_patch_size[1]//2]
    logging.debug('sat_patches.shape: {}'.format(img_sat_patches.shape))
    logging.debug('img_map_patches.shape: {}'.format(img_map_patches.shape))
    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)
logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))
logging.debug('map_patches.shape: {}'.format(map_patches.shape))

map_patches = map_patches.reshape(map_patches.shape + (1,))

# LOADING PREVIOUS WEIGHTS OF MODEL
if net_weights_load:
    logging.info('LOADING PREVIOUS WEIGHTS OF MODEL: {}'.format(net_weights_load))
    model.load_weights(net_weights_load)


# FIT MODEL AND SAVE WEIGHTS
logging.info('FIT MODEL, EPOCHS: {}, SAVE WEIGHTS: {}'.format(epochs, net_weights_dir_save))
# tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(os.path.join(net_weights_dir_save,
                             'w_epoch{epoch:02d}_jaccard{jaccard_coef:.4f}_valjaccard{val_jaccard_coef:.4f}.hdf5'),
                             monitor='val_loss', save_best_only=False)
model.fit(sat_patches, map_patches, epochs=epochs, callbacks=[checkpoint], batch_size=32, validation_split=0.1, verbose=1)