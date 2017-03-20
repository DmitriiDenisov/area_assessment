import cv2
import numpy as np
from keras.callbacks import TensorBoard
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
# from area_assesment.io_operations.visualization import plot_img_mask
from area_assesment.neural_networks.cnn import cnn_v1, cnn_v3

#########################################################
# RUN ON SERVER
# PYTHONPATH=~/area_assesment/ python3 model_train.py
#########################################################

# PATCHING SETTINGS
nn_input_patch_size = (64, 64)
nn_output_patch_size = (16, 16)
step_size = 16

# MODEL TRAINING SETTINGS
epochs = 1
net_weights_version = 63  # use previous weights


# COLLECT PATCHES FROM ALL IMAGES IN THE TRAIN DIRECTORY
dir_train = '../../data/mass_buildings/valid/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tiff')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')

# sat_patches, map_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3)), np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1]))
sat_patches = np.empty((0, nn_input_patch_size[0], nn_input_patch_size[1], 3))
map_patches =  np.empty((0, nn_output_patch_size[0], nn_output_patch_size[1]))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    print('PATCHING IMG: {}/{}, {}, {}'.format(i + 1, len(train_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    print('img_sat.shape: {}'.format(img_sat.shape))
    print('img_map.shape: {}'.format(img_map.shape))
    img_sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    img_map_patches = array2patches(img_map, patch_size=nn_input_patch_size, step_size=step_size)
    img_map_patches = img_map_patches[:,
                      nn_input_patch_size[0]//2 - nn_output_patch_size[0]//2:
                      nn_input_patch_size[0]//2 + nn_output_patch_size[0]//2,
                      nn_input_patch_size[1]//2 - nn_output_patch_size[1]//2:
                      nn_input_patch_size[1]//2 + nn_output_patch_size[1]//2]
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
    print('img_map_patches.shape: {}'.format(img_map_patches.shape))
    # for (sat_patch, map_patch) in list(zip(img_sat_patches, img_map_patches)):
    #     plot_img_mask(sat_patch, map_patch, show_plot=True)

    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)
print('sat_patches.shape: {}'.format(sat_patches.shape))
print('map_patches.shape: {}'.format(map_patches.shape))


# MODEL DETERMINE
model = cnn_v3()
model.summary()

# LOADING PREVIOUS WEIGHTS OF MODEL
model.load_weights('weights/cnn_v3_w_{}.h5'.format(net_weights_version))

# FIT MODEL AND SAVE NEXT
tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
model.fit(sat_patches, map_patches, epochs=epochs, callbacks=[tb_callback])
model.save_weights('weights/cnn_v3_w_{}.h5'.format(net_weights_version+1))

