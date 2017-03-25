import cv2
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import *

# MODEL
model = cnn_v4()
# model.summary()
net_weights_load = '../weights/sakaka_cnn_v4_w_15.h5'
logging.info('LOADING MODEL WEIGHTS: {}'.format(net_weights_load))
model.load_weights(net_weights_load)

# PATCHING SETTINGS
nn_input_patch_size = (64, 64)
step_size = 16
nn_output_patch_size = (16, 16)

# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
# dir_valid = '../../data/mass_buildings/valid/'
dir_valid = '../sakaka_data/valid/'
dir_valid_sat = dir_valid + 'sat/'
dir_valid_map = dir_valid + 'map/'
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')
valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')
output_folder = '../../plots/sakaka_valid/'

for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))):
    print('VALID IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
    img_sat_, img_map_ = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)

    # img_sat_, img_map_ = img_sat_[-900:, -1600:], img_map_[-900:, -1600:]
    print('img_sat_.shape: {}, img_map_.shape: {}'.format(img_sat_.shape, img_map_.shape))
    img_size = img_sat_.shape[:2]
    img_sat = np.empty(((round(img_size[0]/nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1]/nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3))
    img_map = np.empty(((round(img_size[0]/nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1]/nn_input_patch_size[1]) + 2) * nn_input_patch_size[1]))
    img_sat[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]] = img_sat_
    img_map[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]] = img_map_
    print('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    img_sat = img_sat.astype('float32')
    ret, img_map = cv2.threshold(img_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    img_map = img_map.astype('float32')
    img_sat /= 255
    img_map /= 255

    # plot_img_mask(img_sat, img_map)

    sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    print('sat_patches.shape: {}'.format(sat_patches.shape))

    map_patches_pred = model.predict(sat_patches)
    print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot_img_mask(sat_patches[i], map_patches_pred[i], show_plot=True)

    map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_output_patch_size)
    print('map_pred.shape: {}'.format(map_pred.shape))
    map_pred = map_pred[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
                        nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]]
    print('2map_pred.shape: {}'.format(map_pred.shape))

    # PLOT IMG, MASK_TRUE, MASK_PRED
    plot_img_mask_pred(img_sat_, img_map_, map_pred,
                       name='VALID_IMG_{}_stepsize{}'.format(f_sat[len(dir_valid_sat):], step_size),
                       show_plot=False, save_output_path=output_folder)

    # PLOT OVERLAY IMG, MASK_PRED
    # margin_hor = (nn_input_patch_size[0] - nn_output_patch_size[0]) // 2
    # margin_vert = (nn_input_patch_size[1] - nn_output_patch_size[1]) // 2
    plot_img_mask(img_sat_, map_pred,
                  name='VALID_IMG_{}_OVERLAY_stepsize{}'.format(f_sat[len(dir_valid_sat):], step_size),
                  overlay=True, alpha=0.3, show_plot=False, save_output_path=output_folder)

    # plot_img(map_pred,
    #          name='VALID_IMG_{}_PRED_stepsize{}_subpatchsize{}'.format(f_sat, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path=output_folder)

    # np.save('{}.npy'.format(f_sat), map_pred)

    # SCORE EVALUATION
    # score = model.evaluate(sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    ## print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:map_pred.shape[0], :map_pred.shape[1]].reshape(-1), map_pred.reshape(-1))))

