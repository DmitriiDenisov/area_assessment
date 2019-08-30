import cv2
import numpy as np
from keras.callbacks import TensorBoard
from sklearn.feature_extraction.image import *
from sklearn.metrics import jaccard_similarity_score

from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.images_processing.patching import array2patches
from area_assesment.neural_networks.cnn import cnn_v1

# CREATE PATCHES FROM ALL IMAGES
dir_train = '../data/mass_buildings/train/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tiff')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')

sat_patches, map_patches = np.empty((0, 64, 64, 3)), np.empty((0, 64, 64))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files, train_map_files))):
    print('TRAIN, PATCHING IMG: {}/{}, {}, {}'.format(i + 1, len(train_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    print('img_sat.shape: {}'.format(img_sat.shape))
    img_sat_patches, img_map_patches = array2patches(img_sat, step_size=32), array2patches(img_map, step_size=32)
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
    print('img_map_patches.shape: {}'.format(img_map_patches.shape))

    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)

print('sat_patches.shape: {}'.format(sat_patches.shape))
print('map_patches.shape: {}'.format(map_patches.shape))


# MODEL TRAINING
model = cnn_v1()
model.summary()
w_version = 33
model.load_weights('weights/w_{}.h5'.format(w_version))
tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
model.fit(sat_patches, map_patches, epochs=1, callbacks=[tb_callback])
model.save_weights('weights/w_{}.h5'.format(w_version+1))
