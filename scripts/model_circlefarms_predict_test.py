import os
import cv2
import logging
import numpy as np
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import *
from area_assesment.geo.geotiff_utils import write_geotiff
from area_assesment.images_processing.normalization import equalizeHist_rgb
from area_assesment.neural_networks.unet import unet

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])

# MODEL builidings
model = cnn_circle_farms()
model.summary()
net_weights_load = '../weights/cnn_circlefarms/cnn_circlefarms_256to64_epoch22_loss0.0280_valloss0.1006.hdf5'
logging.info('LOADING MODEL WEIGHTS: {}'.format(net_weights_load))
model.load_weights(net_weights_load)

# PATCHING SETTINGS buildings
nn_input_patch_size = (256, 256)
nn_output_patch_size = (64, 64)
subpatch_size = (64, 64)
step_size = 64

# dir_test = os.path.normpath('../sakaka_data/Area_Sakaka_Dawmat_Al_Jandal/')
dir_test = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal/')
output_folder = os.path.normpath('../sakaka_data/circle_farms/output/')

########################################################

pipeline = make_pipeline(StandardScaler(), SGDClassifier(loss='log'))
pipeline.fit(X=img_, y=mask_)


# TEST ON ALL IMAGES IN THE TEST DIRECTORY
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir_test))
valid_sat_files = filenames_in_dir(dir_test, endswith_='.tif')

for i, f_sat in enumerate(valid_sat_files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))

    img_sat_ = cv2.imread(f_sat)
    # dim = (1024, int(img_sat_.shape[0] * (1024.0 / img_sat_.shape[1])))
    # img_sat = cv2.resize(img_sat_, dim, interpolation=cv2.INTER_AREA)

    img_sat = img_sat_.astype('float32')
    img_sat /= 255

    img_size = img_sat.shape[:2]
    img_sat_upscale = np.empty(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                                (round(img_size[1] / nn_input_patch_size[1]) + 1) * nn_input_patch_size[1], 3))
    img_sat_upscale[:, :, 0] = img_sat[:, :, 0].mean()*2
    img_sat_upscale[:, :, 1] = img_sat[:, :, 1].mean()*2
    img_sat_upscale[:, :, 2] = img_sat[:, :, 2].mean()*2

    img_sat_upscale[nn_input_patch_size[0]//2: nn_input_patch_size[0]//2 + img_size[0],
                    nn_input_patch_size[1]//2:nn_input_patch_size[1]//2 + img_size[1]] = img_sat

    logging.info('raw img_sat_.shape: {}'.format(img_sat_.shape))
    logging.info('img_sat_upscale.shape: {}'.format(img_sat_upscale.shape))

    sat_patches = array2patches(img_sat_upscale, patch_size=nn_input_patch_size, step_size=step_size)
    logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))

    logging.info('PREDICTING, sat_patches.shape:{}'.format(sat_patches.shape))
    map_patches_pred = model.predict(sat_patches)
    logging.debug('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # map_patches_pred = np.array([np.argmax(map_patches_pred[k], axis=2) for k in range(map_patches_pred.shape[0])])
    # map_patches_pred = map_patches_pred[:, :, :, 0]
    logging.debug('2 map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot_img_mask(sat_patches[i], map_patches_pred[i], show_plot=True)

    # Extracting subpatches
    map_patches_pred = map_patches_pred[:,
                                        nn_output_patch_size[0] // 2 - subpatch_size[0] // 2:
                                        nn_output_patch_size[0] // 2 + subpatch_size[0] // 2,
                                        nn_output_patch_size[1] // 2 - subpatch_size[1] // 2:
                                        nn_output_patch_size[1] // 2 + subpatch_size[1] // 2]

    map_pred = patches2array(map_patches_pred, img_size=img_sat_upscale.shape[:2], step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=subpatch_size)
    logging.info('map_pred.shape: {}'.format(map_pred.shape))

    map_pred = map_pred[nn_input_patch_size[0]//2: nn_input_patch_size[0]//2 + img_size[0],
                        nn_input_patch_size[1]//2:nn_input_patch_size[1]//2 + img_size[1]]
    logging.info('raw (imgsize) map_pred.shape: {}'.format(map_pred.shape))

    # map_pred = cv2.resize(map_pred, img_sat_.shape[:2][::-1], interpolation=cv2.INTER_AREA)

    logging.info('upscale raw (imgsize) map_pred.shape: {}'.format(map_pred.shape))

    # PLOT OVERLAY IMG, MASK_PRED
    # plot_img_mask(img_sat_, map_pred, name='{}_OVERLAY_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size),
    #               overlay=True, alpha=0.3, show_plot=True, save_output_path=None)

    # plot_img(map_pred, name='{}_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path=output_folder)

    # plot_img(img_sat_, name='{}_ORIG_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path=output_folder)

    # WRITE GEOTIFF
    geotiff_output_path = os.path.join(output_folder, '{}_HEATMAP.tif'.format(f_sat.split('/')[-1][:-4]))
    write_geotiff(geotiff_output_path, raster_layers=(map_pred*255).astype('int'), gdal_ds=f_sat)

    # np.save('{}_OVERLAY_GRABCUT_stepsize{}.npy'.format(f_sat.split('/')[-1][:-4], step_size), map_pred)
