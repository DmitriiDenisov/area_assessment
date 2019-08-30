import os
import cv2
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import *
from area_assesment.geo.geotiff_utils import write_geotiff


logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])

# MODEL
model = cnn_v4_cont()
# model.summary()
net_weights_load = '../weights/cnn_v4_cont/w_epoch04_jaccard0.3865_valjaccard0.1452.hdf5'
logging.info('LOADING MODEL WEIGHTS: {}'.format(net_weights_load))
model.load_weights(net_weights_load)

# PATCHING SETTINGS
nn_input_patch_size = (64, 64)
step_size = 16
nn_output_patch_size = (16, 16)


# TEST ON ALL IMAGES IN THE TEST DIRECTORY
# dir_valid = os.path.normpath('../sakaka_data/test/')  # '../../data/mass_buildings/valid/'
# dir_valid_sat = os.path.join(dir_valid, 'sat/')
dir_valid_sat = os.path.normpath('/storage/_pdata/sakaka/train/sat')
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir_valid_sat))
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')[1:]
output_folder = '../sakaka_data/output/sakaka_test/'

for i, f_sat in enumerate(valid_sat_files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))
    img_sat_ = cv2.imread(f_sat)  # [-500:, -1000:]

    logging.debug('raw img_sat_.shape: {}'.format(img_sat_.shape))
    img_size = img_sat_.shape[:2]
    img_sat = np.empty(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1] / nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3))
    img_sat[nn_input_patch_size[0]:nn_input_patch_size[0] + img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1] + img_size[1]] = img_sat_
    img_sat = img_sat.astype(np.uint8)
    img_sat = img_sat[:, :, 0].reshape((img_sat.shape[0], img_sat.shape[1], 1))
    # cont = cv2.findContours(cv2.threshold(cv2.cvtColor(img_sat, cv2.COLOR_BGR2GRAY),
    #                                       127, 255, cv2.THRESH_BINARY)[1], 1, 2)[1]
    # cont = cv2.drawContours(np.zeros(img_sat.shape[:2], np.uint8), cont, -1, 255, 1)
    # img_sat = np.append(img_sat, cont.reshape((cont.shape[0], cont.shape[1], 1)), 2)

    logging.debug('oversized img_sat.shape: {}'.format(img_sat.shape))
    img_sat = img_sat.astype('float32')
    img_sat = np.exp(img_sat / img_sat.max())

    sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))

    logging.info('PREDICTING, sat_patches.shape:{}'.format(sat_patches.shape))
    map_patches_pred = model.predict(sat_patches)
    logging.debug('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot_img_mask(sat_patches[i], map_patches_pred[i], show_plot=True)

    map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_output_patch_size)
    logging.debug('map_pred.shape: {}'.format(map_pred.shape))
    map_pred = map_pred[nn_input_patch_size[0]:nn_input_patch_size[0] + img_size[0],
                        nn_input_patch_size[1]:nn_input_patch_size[1] + img_size[1]]
    logging.debug('raw (imgsize) map_pred.shape: {}'.format(map_pred.shape))

    # PLOT OVERLAY IMG, MASK_PRED
    plot_img_mask(img_sat_, map_pred, name='{}_OVERLAY_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size),
                  overlay=True, alpha=0.5, show_plot=False, save_output_path=output_folder)

    plot_img(map_pred, name='{}_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
             show_plot=False, save_output_path=output_folder)

    plot_img(img_sat_, name='{}_ORIG_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
             show_plot=False, save_output_path=output_folder)

    # WRITE GEOTIFF
    # geotiff_output_path = output_folder+'{}_HEATMAP.tif'.format(f_sat.split('/')[-1][:-4])
    # write_geotiff(geotiff_output_path, raster_layers=(map_pred*255).astype('int'), gdal_ds=f_sat)

    # np.save('{}_OVERLAY_GRABCUT_stepsize{}.npy'.format(f_sat.split('/')[-1][:-4], step_size), map_pred)
