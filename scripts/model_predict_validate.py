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

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])

# MODEL
model = cnn_v6()
# model.summary()
net_weights_load = '../weights/cnn_v6/weights_epoch12_loss0.0555.hdf5'
logging.info('LOADING MODEL WEIGHTS: {}'.format(net_weights_load))
model.load_weights(net_weights_load)

# PATCHING SETTINGS
nn_input_patch_size = (64, 64)
step_size = 2
nn_output_patch_size = (16, 16)

# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
# dir_valid = '../../data/mass_buildings/valid/'
dir_valid = '../sakaka_data/valid/'
dir_valid_sat = dir_valid + 'sat/'
dir_valid_map = dir_valid + 'map/'
logging.info('VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY: {}, {}'.format(dir_valid_sat, dir_valid_map))
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')
valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')
output_folder = '../sakaka_data/output/sakaka_valid/'

for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))):
    logging.info('LOADING IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
    img_sat_, img_map_ = cv2.imread(f_sat), cv2.imread(f_map, cv2.IMREAD_GRAYSCALE)
    img_sat_, img_map_ = img_sat_[255:, 128:], img_map_[255:, 128:]
    logging.debug('img_sat_.shape: {}, img_map_.shape: {}'.format(img_sat_.shape, img_map_.shape))

    img_size = img_sat_.shape[:2]
    img_sat = np.empty(((round(img_size[0]/nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1]/nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3))
    img_map = np.empty(((round(img_size[0]/nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1]/nn_input_patch_size[1]) + 2) * nn_input_patch_size[1]))
    img_sat[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]] = img_sat_
    img_map[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]] = img_map_
    logging.debug('img_sat.shape: {}, img_map.shape: {}'.format(img_sat.shape, img_map.shape))
    img_sat = img_sat.astype('float32')
    ret, img_map = cv2.threshold(img_map.astype(np.uint8), 127, 255, cv2.THRESH_BINARY)
    img_map = img_map.astype('float32')
    img_sat /= 255
    img_map /= 255

    # plot_img_mask(img_sat, img_map)

    sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))

    logging.info('PREDICTING')
    map_patches_pred = model.predict(sat_patches)
    logging.debug('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot_img_mask(sat_patches[i], map_patches_pred[i], show_plot=True)

    map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_output_patch_size)
    logging.debug('map_pred.shape: {}'.format(map_pred.shape))

    map_pred = map_pred[nn_input_patch_size[0]:nn_input_patch_size[0]+img_size[0],
                        nn_input_patch_size[1]:nn_input_patch_size[1]+img_size[1]]
    logging.debug('2map_pred.shape: {}'.format(map_pred.shape))

    # PLOT IMG, MASK_TRUE, MASK_PRED
    plot_img_mask_pred(img_sat_, img_map_, map_pred,
                       name='{}_COMPARISON_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size),
                       show_plot=True, save_output_path=output_folder)

    # PLOT OVERLAY IMG, MASK_PRED
    plot_img_mask(img_sat_, map_pred,
                  name='{}_OVERLAY_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size),
                  overlay=True, alpha=0.5, show_plot=False, save_output_path=output_folder)

    # plot_img(map_pred,
    #          name='VALID_IMG_{}_PRED_stepsize{}_subpatchsize{}'.format(f_sat, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path=output_folder)

    # np.save('{}.npy'.format(f_sat), map_pred)

    # SCORE EVALUATION
    # score = model.evaluate(sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    ## print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:map_pred.shape[0], :map_pred.shape[1]].reshape(-1), map_pred.reshape(-1))))

