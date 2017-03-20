import cv2
import numpy as np
import matplotlib
from area_assesment.images_processing.polygons import mask_to_polygons
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import cnn_v1, cnn_v3

# MODEL
model = cnn_v3()
# model.summary()
model.load_weights('weights/cnn_v3_w_63.h5')


# PATCHING SETTINGS
nn_input_patch_size = (64, 64)  # how to patching validation image: size of input layer of MODEL
step_size = 16
nn_output_patch_size = (16, 16)  # size of central sub-patch of predicted patch, to transform predicted patches into image


# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
dir_valid = '../../data/mass_buildings/valid/'
dir_valid_sat = dir_valid + 'sat/'
dir_valid_map = dir_valid + 'map/'
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tiff')
valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')
for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))[1:]):
    print('VALID IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    img_sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))

    map_patches_pred = model.predict(img_sat_patches)
    print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    img_map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                                 nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_output_patch_size)
    print('img_map_pred.shape: {}'.format(img_map_pred.shape))

    # for i in range(img_sat_patches.shape[0]):
    #     plot_img_mask(img_sat_patches[i], map_patches_pred[i], show_plot=True)

    # PLOT IMG, MASK_TRUE, MASK_PRED
    # plot_img_mask_pred(img_sat, img_map, img_map_pred,
    #                    name='3_cnn_v3_VALID_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #                    show_plot=False, save_output_path='../../plots/')

    plot_img(img_sat,
             name='5_cnn_v3_VALID_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
             show_plot=False, save_output_path='../../plots/')

    plot_img(img_map,
             name='7_cnn_v3_VALID_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
             show_plot=False, save_output_path='../../plots/')

    plot_img(img_map_pred, name='8_cnn_v3_VALID_IMG{}_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
             show_plot=False, save_output_path='../../plots/')

    polygons = mask_to_polygons(img_map_pred)
    print(polygons)
    break

    # SCORE EVALUATION
    # score = model.evaluate(img_sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    ## print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:img_map_pred.shape[0], :img_map_pred.shape[1]].reshape(-1), img_map_pred.reshape(-1))))

