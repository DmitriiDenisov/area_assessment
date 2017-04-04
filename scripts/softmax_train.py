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
from area_assesment.io_operations.visualization import plot2, plot3
from area_assesment.neural_networks.cnn import *
from area_assesment.neural_networks.unet import *
import hashlib

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 model_train.py
#########################################################


logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.DEBUG,
                    handlers=[logging.StreamHandler()])

# MODEL DEFINITION
logging.info('MODEL DEFINITION')
model = unet(128, 128, 3)  # unet_circle_farms(256, 256, 3) #unet(64, 64, 3)
model.summary()

# PATCHING SETTINGS
nn_input_patch_size = (128, 128)
nn_output_patch_size = nn_input_patch_size
# step_size = 16
patches_per_img = 100

# MODEL SETTINGS
epochs = 50
net_weights_load = os.path.normpath('../weights/unet/buildings-unet_128x128_epoch49_iu0.9184_val_iu0.9389.hdf5')
net_weights_dir_save = os.path.normpath('../weights/unet/')
########################################################

# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir_train = os.path.normpath('../sakaka_data/buildings/Dawmat_Al_Jandal/')
dir_train_sat = os.path.join(dir_train, 'sat/')
dir_train_map = os.path.join(dir_train, 'map/')
logging.info('COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY: {}, {}'.format(dir_train_sat, dir_train_map))
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tif')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')

sat_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3))
map_patches = np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1], 2))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    logging.info('LOADING IMG: {}/{}, {}, {}'.format(i + 1, len(train_map_files), f_sat, f_map))
    img_sat_, img_map_ = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)
    logging.debug('img_sat_.shape: {}, img_map_.shape: {}'.format(img_sat_.shape, img_map_.shape))
    # logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    img_sat = img_sat_.astype('float32')
    ret, img_map = cv2.threshold(img_map_.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    img_map = img_map.astype('float32')

    img_sat /= 255
    img_map /= 255
    img_map_1 = img_map.copy()
    img_map_2 = np.ones(img_map.shape) - img_map.copy()
    # plot2(img_map_1, img_map_2, show_plot=True)

    img_sat_patches = extract_patches_2d(img_sat, nn_input_patch_size, max_patches=patches_per_img, random_state=3)
    img_map_1_patches = extract_patches_2d(img_map_1, nn_input_patch_size, max_patches=patches_per_img, random_state=3)
    img_map_2_patches = extract_patches_2d(img_map_2, nn_input_patch_size, max_patches=patches_per_img, random_state=3)

    # for (sat_patch, map_1_patch, map_2_patch) in list(zip(img_sat_patches, img_map_1_patches, img_map_2_patches)):
    #     logging.debug(sat_patch.shape, map_1_patch.shape)
    #     plot3(sat_patch, map_1_patch, map_2_patch, show_plot=True)

    # print(img_map_1_patches.shape, img_map_2_patches.shape)
    img_map_patches = np.array([[img_map_1_patches[k], img_map_2_patches[k]] for k in range(patches_per_img)])
    img_map_patches = np.moveaxis(img_map_patches, 1, -1)

    logging.debug('sat_patches.shape: {}'.format(img_sat_patches.shape))
    logging.debug('img_map_patches.shape: {}'.format(img_map_patches.shape))

    # for i in range(img_map_patches.shape[0]):
    #     logging.debug(img_sat_patches[i], img_map_patches[i].shape)
    #     plot3(img_sat_patches[i], img_map_patches[i, :, :, 0], img_map_patches[i, :, :, 1], show_plot=True)

    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)
logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))
logging.debug('map_patches.shape: {}'.format(map_patches.shape))


# LOADING PREVIOUS WEIGHTS OF MODEL
if net_weights_load:
    logging.info('LOADING PREVIOUS WEIGHTS OF MODEL: {}'.format(net_weights_load))
    model.load_weights(net_weights_load)


# FIT MODEL AND SAVE WEIGHTS
logging.info('FIT MODEL, EPOCHS: {}, SAVE WEIGHTS: {}'.format(epochs, net_weights_dir_save))
# tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
checkpoint = ModelCheckpoint(os.path.join(net_weights_dir_save,
                             'buildings-unet_128x128x3_epoch3{epoch:02d}_iu{jaccard_coef:.4f}_val_iu{val_jaccard_coef:.4f}.hdf5'),
                             monitor='val_loss', save_best_only=False)
model.fit(sat_patches, map_patches, epochs=epochs, callbacks=[checkpoint], batch_size=128, validation_split=0.1)
