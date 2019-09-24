import os
from os.path import join, isfile
import cv2
import logging
import numpy as np

from keras_preprocessing.image import load_img

from area_assesment.geo.geotiff_utils import write_geotiff
# from area_assesment.geo.utils import write_geotiff
from area_assesment.images_processing.patching import array2patches_new, patches2array_overlap_norm
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import save_mask_and_im_overlayer, \
    just_show_numpy_as_image

from tensorflow.keras.models import load_model
from area_assesment.neural_networks.metrics import fmeasure, precision, recall, jaccard_coef, dice_coef_K

logging.getLogger().setLevel(logging.INFO)

MODEL_PATH = '../weights/unet_mecca/z18_model/z18_epoch166_jaccard0.885_dice_coef_K0.939_fmeasure0.958.hdf5'
# MODEL_PATH = '../weights/unet/buildings-unet_64x64x3_epoch712_iu0.9133_val_iu0.9472.hdf5'
# MODEL_PATH = '../weights/unet_mecca/good_models/retrained_10sept_w_epoch01_jaccard0.9152.hdf5'
# MODEL_PATH = '../weights/unet_mecca/try_128x128_unet_old_with_lambda_layer/w_epoch45_jaccard0.889_dice_coef_K0.941_fmeasure0.959.hdf5'
# MODEL_PATH = '../weights/unet_mecca/try_128x128_unet_old_without_lambda_layer/w_no_lambda_epoch147_jaccard0.942_dice_coef_K0.970_fmeasure0.979.hdf5'
# MODEL_PATH = '../weights/unet_mecca/two_inputs_epoch25_jaccard0.421_dice_coef_K0.598_fmeasure0.726.hdf5'
print('Loading model from {}'.format(MODEL_PATH))
model = load_model(MODEL_PATH,
                   custom_objects={
                       "precision": precision,
                       "recall": recall,
                       "fmeasure": fmeasure,
                       "jaccard_coef": jaccard_coef,
                       "dice_coef_K": dice_coef_K
                   }
                   )
model.summary()

# PATCHING SETTINGS buildings
nn_input_patch_size = model.output_shape[1:3]
step_size = 64
if len(model.input_shape) == 2:
    mode = 'two_inputs'
else:
    mode = 'one_input'

# dir_test = os.path.normpath('../sakaka_data/buildings/valid/sat/')  # '../../data/mass_buildings/valid/'
dir_nokia = os.path.normpath('../data/val/nokia_mask')
dir_test = os.path.normpath('../data/val/sat_2/z18')  # '../../data/mass_buildings/valid/'
files_names = sorted([f for f in os.listdir(dir_test) if isfile(join(dir_test, f))])
# dir_test = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal/')
# output_folder = os.path.normpath('../sakaka_data/buildings/output/buildings_unet_128x128_epoch446_subpatch64_stepsize64/')
output_folder = os.path.normpath('../output_data/Mecca_old_model')
########################################################

# TEST ON ALL IMAGES IN THE TEST DIRECTORY
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir_test))
valid_sat_files = filenames_in_dir(dir_test, endswith_='.tif')
for i, f_sat in enumerate(files_names):  # valid_sat_files
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))
    img_sat = cv2.imread(join(dir_test, f_sat))  # [2000:2256, 0:256]
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    # plot1(img_sat_, show_plot=True)
    img_size = img_sat.shape[:2]

    # UPSCALE SO SLICING WINDOW WILL FIT IN FULL IMAGE
    img_sat_upscale = np.zeros(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                                (round(img_size[1] / nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3),
                               dtype=np.float32)
    img_upscale_size = img_sat_upscale.shape[:2]
    img_sat_upscale[:, :, 0] = img_sat[:, :, 0].mean()
    img_sat_upscale[:, :, 1] = img_sat[:, :, 1].mean()
    img_sat_upscale[:, :, 2] = img_sat[:, :, 2].mean()
    img_sat_upscale[0: img_size[0], 0:img_size[1]] = img_sat

    if mode == 'two_inputs':
        img_nokia = (np.array(
            load_img(join(dir_nokia, '{}_nokia_MASK.tif'.format(f_sat[:-4])), grayscale=True)) / 255).astype(np.uint8)
        img_sat_upscale_nokia = np.zeros(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                                          (round(img_size[1] / nn_input_patch_size[1]) + 2) * nn_input_patch_size[1]),
                                         dtype=np.float32)
        img_sat_upscale_nokia[:, :] = 0
        img_sat_upscale_nokia[0: img_size[0], 0:img_size[1]] = img_nokia
        just_show_numpy_as_image((img_sat_upscale_nokia * 255).astype(np.uint8), type='black', name='test.tif')
        # 1 / 0

    logging.debug('img_sat.shape: {}'.format(img_sat.shape))
    logging.debug('img_sat_upscale.shape: {}'.format(img_sat_upscale.shape))

    # del img_sat

    # GET PATCHES:
    # sat_patches_old = array2patches(img_sat_upscale, patch_size=nn_input_patch_size, step_size=step_size)
    sat_patches = array2patches_new(img_sat_upscale, patch_size=nn_input_patch_size, step_size=step_size)
    if mode == 'two_inputs':
        nokia_patches = array2patches_new(img_sat_upscale_nokia, patch_size=nn_input_patch_size, step_size=step_size)
        del img_sat_upscale_nokia
    del img_sat_upscale
    logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))

    logging.info('PREDICTING, sat_patches.shape:{}'.format(sat_patches.shape))
    if mode == 'two_inputs':
        map_patches_pred = model.predict([sat_patches, nokia_patches.reshape(nokia_patches.shape + (1,))], verbose=1)
        del nokia_patches, sat_patches
    else:
        map_patches_pred = model.predict(sat_patches, verbose=1)
        del sat_patches

    logging.debug('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    if len(model.output_shape) == 4:
        map_patches_pred = map_patches_pred[:, :, :, 0]
    logging.debug('2 map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # PATCHES BACK TO ARRAY:
    # map_pred = patches2array(map_patches_pred, img_size=img_upscale_size, step_size=step_size,
    #                        nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=nn_input_patch_size)
    map_pred = patches2array_overlap_norm(map_patches_pred, img_upscale_size, img_upscale_size,
                                          patch_size=nn_input_patch_size, step_size=step_size)
    # map_pred = patches2array_overlap(map_patches_pred, img_size=img_upscale_size, step_size=step_size,
    #                         patch_size=nn_input_patch_size, subpatch_size=nn_input_patch_size)
    logging.debug('map_pred.shape: {}'.format(map_pred.shape))

    map_pred = map_pred[0:img_size[0], 0:img_size[1]]
    logging.debug('raw (imgsize) map_pred.shape: {}'.format(map_pred.shape))

    if True:
        save_mask_and_im_overlayer(img_sat, map_pred, save_output_path='overlay_test_{}.tif'.format(i))
        map_pred = (map_pred * 255).astype(np.uint8)
        map_pred[map_pred < 76] = 0
        map_pred[map_pred >= 76] = 255

        just_show_numpy_as_image(map_pred, type='Black', name='map_pred_{}.tif'.format(i))
        # 1 / 0

    # WRITE GEOTIFF
    geotiff_output_path = os.path.join(output_folder, '{}_HEATMAP.tif'.format(os.path.basename(f_sat)[:-4]))
    write_geotiff(geotiff_output_path, raster_layers=(map_pred * 255).astype(np.int16), gdal_ds=join(dir_test, f_sat))
    del map_pred
