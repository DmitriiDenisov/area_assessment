import cv2
import numpy as np
import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array2
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask
from area_assesment.neural_networks.cnn import cnn_v1, cnn_v2

# MODEL
model = cnn_v1()
# model.summary()
model.load_weights('weights/w_34.h5')


# PATCHING SETTINGS
patch_size = (64, 64)  # how to patching validation image: size of input layer of MODEL
step_size = 8
subpatch_size = (32, 32)  # size of central sub-patch of predicted patch, to transform predicted patches into image


# TEST (WITHOUT LABELED IMAGE)
dir_test = '../../sakaka_data/'
dir_test_sat = dir_test + 'sat/'
test_sat_files = filenames_in_dir(dir_test_sat, endswith_='.tif')
for i, f_sat in enumerate(test_sat_files):
    print('TEST IMG: {}/{}, {}'.format(i+1, len(test_sat_files), f_sat))
    img_sat = plt.imread(f_sat)[:1500, :1500]
    img_sat = img_sat.astype('float32')
    img_sat /= 255
    print('img_sat.shape: {}'.format(img_sat.shape))

    img_sat_patches = array2patches(img_sat, patch_size=patch_size, step_size=step_size)
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))

    map_patches_pred = model.predict(img_sat_patches)
    print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))
    img_map_pred = patches2array_overlap(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                                         subpatch_size=subpatch_size)
    # img_map_pred = patches2array2(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size)
    print('img_map_pred.shape: {}'.format(img_map_pred.shape))

    plot_img_mask(img_sat, img_map_pred,
                  name='TEST_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, subpatch_size[0]),
                  show_plot=False, save_output_path='../../plots/')
