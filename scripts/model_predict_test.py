import os
import cv2
import gdal
import numpy as np
import matplotlib
matplotlib.use('Agg')
# import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import *
from area_assesment.geo.geotiff_utils import write_geotiff


# MODEL
model = cnn_v4()
# model.summary()
model.load_weights('../weights/sakaka_cnn_v4_w_13.h5')


# PATCHING SETTINGS
nn_input_patch_size = (64, 64)  # how to patching validation image: size of input layer of MODEL
step_size = 16
nn_output_patch_size = (16, 16)  # size of central sub-patch of predicted patch, to transform predicted patches into image


# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
# dir_valid = '../../data/mass_buildings/valid/'  # '../../sakaka_data/train/'  # '../../data/mass_buildings/valid/'
# dir_valid = '../sakaka_data/test/'
dir_valid_sat = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Er_Riadh/')  #dir_valid + 'sat/'
# dir_valid_sat = dir_valid + 'sat/'
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')
output_folder = '../sakaka_data/output/sakaka_test/'

for i, f_sat in enumerate(valid_sat_files):
    print('VALID IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))
    img_sat_ = cv2.imread(f_sat)  # [-1000:, -1000:]

    print('raw img_sat_.shape: {}'.format(img_sat_.shape))
    img_size = img_sat_.shape[:2]
    img_sat = np.empty(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                        (round(img_size[1] / nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3))
    img_sat[nn_input_patch_size[0]:nn_input_patch_size[0] + img_size[0],
            nn_input_patch_size[1]:nn_input_patch_size[1] + img_size[1]] = img_sat_

    print('oversized img_sat.shape: {}'.format(img_sat.shape))
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    sat_patches = array2patches(img_sat, patch_size=nn_input_patch_size, step_size=step_size)
    print('sat_patches.shape: {}'.format(sat_patches.shape))

    map_patches_pred = model.predict(sat_patches)
    print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot_img_mask(sat_patches[i], map_patches_pred[i], show_plot=True)

    map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_output_patch_size)
    print('map_pred.shape: {}'.format(map_pred.shape))
    map_pred = map_pred[nn_input_patch_size[0]:nn_input_patch_size[0] + img_size[0],
                        nn_input_patch_size[1]:nn_input_patch_size[1] + img_size[1]]
    print('2imgsize map_pred.shape: {}'.format(map_pred.shape))

    # PLOT OVERLAY IMG, MASK_PRED
    plot_img_mask(img_sat_, map_pred,
                  name='TEST_IMG_{}_OVERLAY_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size),
                  overlay=True, alpha=0.3, show_plot=False, save_output_path=output_folder)

    # plot_img(map_pred,
    #          name='VALID_IMG_{}_PRED_stepsize{}_subpatchsize{}'.format(f_sat, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path=output_folder)

    print('write_geotiff')
    write_geotiff(output_folder+'{}_HEATMAP.tif'.format(f_sat.split('/')[-1][:-4]),
                  raster_layers=(map_pred*255).astype('int'), gdal_ds=gdal.Open(f_sat))

