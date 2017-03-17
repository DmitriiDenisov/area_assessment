import cv2
import numpy as np
from area_assesment.images_processing.patching import array2patches, patches2array_overlap
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred
from area_assesment.neural_networks.cnn import cnn_v1


# MODEL
model = cnn_v1()
# model.summary()
model.load_weights('weights/w_33.h5')


# PATCHING SETTINGS
patch_size = (64, 64)  # how to patching validation image: size of input layer of MODEL
step_size = 16
subpatch_size = (32, 32)  # size of central sub-patch of predicted patch, to transform predicted patches into image


# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
dir_valid = '../../data/mass_buildings/valid/'
dir_valid_sat = dir_valid + 'sat/'
dir_valid_map = dir_valid + 'map/'
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tiff')
valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')
for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))):
    print('VALID IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    img_sat_patches = array2patches(img_sat, patch_size=patch_size, step_size=step_size)
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))

    map_patches_pred = model.predict(img_sat_patches)
    print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    img_map_pred = patches2array_overlap(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                                         subpatch_size=subpatch_size)
    print('img_map_pred.shape: {}'.format(img_map_pred.shape))

    # PLOT IMG, MASK_TRUE, MASK_PRED
    print('PLOT IMG, MASK_TRUE, MASK_PRED'.format(img_map_pred.shape))
    plot_img_mask_pred(img_sat, img_map, img_map_pred,
                       name='4_VALID_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, subpatch_size[0]),
                       show_plot=False, save_output_path='../../plots/')

    # SCORE EVALUATION
    # score = model.evaluate(img_sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    ## print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:img_map_pred.shape[0], :img_map_pred.shape[1]].reshape(-1), img_map_pred.reshape(-1))))


# TEST (WITHOUT LABELED IMAGE)
# dir_test = '../../data/mass_buildings/test/'
# dir_test_sat = dir_test + 'sat/'
# test_sat_files = filenames_in_dir(dir_test_sat, endswith_='.tiff')
# for i, f_sat in enumerate(test_sat_files):
#     print('TEST IMG: {} ({}), {}'.format(i+1, len(test_sat_files), f_sat))
#     # img_sat = plt.imread(f_sat)
#     img_sat = plt.imread(f_sat)
#     img_sat = img_sat.astype('float32')
#     img_sat /= 255
#     print('img_sat.shape: {}'.format(img_sat.shape))
#
#     img_sat_patches = array2patches(img_sat)
#     print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
#
#     map_patches_pred = model.predict(img_sat_patches)
#     print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))
#     img_map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2])
#     print('img_map_pred.shape: {}'.format(img_map_pred.shape))
#
#     plot_img_mask(img_sat, img_map_pred)
