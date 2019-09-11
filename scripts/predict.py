import os
from os.path import join

import cv2
import matplotlib.pyplot as plt
import logging

# from tqdm import tqdm

from area_assesment.images_processing.patching import array2patches, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot3, plot2, plot1, save_mask_and_im_overlayer
from area_assesment.neural_networks.cnn import *
from area_assesment.geo.utils import write_geotiff

from keras.models import load_model
from area_assesment.neural_networks.metrics import fmeasure, precision, recall, jaccard_coef


# print(keras.__version__)

def cut_mask(mask, step_size):
    counter = 64 // step_size
    temp_mask = mask[64-step_size:, 64-step_size:]
    return temp_mask / (counter ** 2)


logging.getLogger().setLevel(logging.INFO)


# MODEL_PATH = '../weights/unet/buildings-unet_64x64x3_epoch712_iu0.9133_val_iu0.9472.hdf5'
# MODEL_PATH = '../weights/unet_mecca/w_epoch32_jaccard0.9272.hdf5'
MODEL_PATH = '../weights/unet_mecca/good_models/retrained_10sept_w_epoch01_jaccard0.9152.hdf5'
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


# PATCHING SETTINGS buildings
nn_input_patch_size = (64, 64)
nn_output_patch_size = nn_input_patch_size
subpatch_size = (64, 64)
step_size = 64

# dir_test = os.path.normpath('../sakaka_data/buildings/valid/sat/')  # '../../data/mass_buildings/valid/'
dir_test = os.path.normpath('../data/test/sat_2')  # '../../data/mass_buildings/valid/'
# dir_test = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal/')
# output_folder = os.path.normpath('../sakaka_data/buildings/output/buildings_unet_128x128_epoch446_subpatch64_stepsize64/')
output_folder = os.path.normpath('../output_data/Mecca')
########################################################

# TEST ON ALL IMAGES IN THE TEST DIRECTORY
logging.info('TEST ON ALL IMAGES IN THE TEST DIRECTORY: {}'.format(dir_test))
valid_sat_files = filenames_in_dir(dir_test, endswith_='.tif')
for i, f_sat in enumerate(valid_sat_files):
    logging.info('LOADING IMG: {}/{}, {}'.format(i + 1, len(valid_sat_files), f_sat))
    img_sat = cv2.imread(f_sat)  # [2000:2256, 0:256]
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    # plot1(img_sat_, show_plot=True)

    img_size = img_sat.shape[:2]

    # Делаем апскейл для того, чтобы окошко 64x64 вместилось
    img_sat_upscale = np.zeros(((round(img_size[0] / nn_input_patch_size[0]) + 2) * nn_input_patch_size[0],
                                (round(img_size[1] / nn_input_patch_size[1]) + 2) * nn_input_patch_size[1], 3), dtype=np.float32)
    img_upscale_size = img_sat_upscale.shape[:2]
    img_sat_upscale[:, :, 0] = img_sat[:, :, 0].mean()
    img_sat_upscale[:, :, 1] = img_sat[:, :, 1].mean()
    img_sat_upscale[:, :, 2] = img_sat[:, :, 2].mean()

    img_sat_upscale[nn_input_patch_size[0] - subpatch_size[0]: nn_input_patch_size[0] - subpatch_size[0] + img_size[0],
    nn_input_patch_size[1] - subpatch_size[1]:nn_input_patch_size[1] - subpatch_size[1] + img_size[1]] = img_sat

    logging.debug('img_sat.shape: {}'.format(img_sat.shape))
    logging.debug('img_sat_upscale.shape: {}'.format(img_sat_upscale.shape))

    del img_sat

    sat_patches = array2patches(img_sat_upscale, patch_size=nn_input_patch_size, step_size=step_size)
    del img_sat_upscale
    logging.debug('sat_patches.shape: {}'.format(sat_patches.shape))

    logging.info('PREDICTING, sat_patches.shape:{}'.format(sat_patches.shape))
    map_patches_pred = model.predict(sat_patches, verbose=1)
    del sat_patches

    logging.debug('map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # map_patches_pred = np.array([np.argmax(map_patches_pred[k], axis=2) for k in range(map_patches_pred.shape[0])])
    map_patches_pred = map_patches_pred[:, :, :, 0]
    logging.debug('2 map_patches_pred.shape: {}'.format(map_patches_pred.shape))

    # for i in range(sat_patches.shape[0]):
    #     plot2(sat_patches[i], map_patches_pred[i], show_plot=True)

    # Extracting subpatches
    map_patches_pred = map_patches_pred[:,
                       nn_output_patch_size[0] // 2 - subpatch_size[0] // 2:
                       nn_output_patch_size[0] // 2 + subpatch_size[0] // 2,
                       nn_output_patch_size[1] // 2 - subpatch_size[1] // 2:
                       nn_output_patch_size[1] // 2 + subpatch_size[1] // 2]

    map_pred = patches2array(map_patches_pred, img_size=img_upscale_size, step_size=step_size,
                             nn_input_patch_size=nn_input_patch_size, nn_output_patch_size=subpatch_size)

    logging.debug('map_pred.shape: {}'.format(map_pred.shape))

    map_pred = map_pred[
               nn_input_patch_size[0] - subpatch_size[0]: nn_input_patch_size[0] - subpatch_size[0] + img_size[0],
               nn_input_patch_size[1] - subpatch_size[1]:nn_input_patch_size[1] - subpatch_size[1] + img_size[1]]
    logging.debug('raw (imgsize) map_pred.shape: {}'.format(map_pred.shape))

    # npy_output_path = os.path.join(output_folder, '{}_NUMPY.npy'.format(os.path.basename(f_sat)[:-4]))
    # np.save(npy_output_path, map_pred)

    # map_pred = cut_mask(map_pred, step_size)

    # 1/0

    # fig, ax = plt.subplots(ncols=4, nrows=1, figsize=(10, 10))
    # ax[0].imshow(img_sat_), ax[0].set_title('1'), ax[0].axis('off')
    # ax[1].matshow(map_pred**2), ax[1].set_title('2'), ax[1].axis('off')
    # ax[2].matshow(map_pred ** 3), ax[1].set_title('3'), ax[2].axis('off')
    # ax[3].matshow(map_pred ** 4), ax[1].set_title('4'), ax[3].axis('off')
    # plt.show()


    if False:
        # PLOT OVERLAY IMG, MASK_PRED
        #plot2((img_sat_*255).astype(np.uint8), (map_pred*255).astype(np.uint8), name='{}_OVERLAY_HEATMAP_stepsize{}_subpatchsize{}'.format(os.path.basename(f_sat)[:-5],
        #                                                                                     step_size, subpatch_size[0]),
        #      overlay=True, alpha=0.5, show_plot=False, save_output_path=output_folder)
        save_mask_and_im_overlayer(img_sat, map_pred, save_output_path='overlay_test_{}.tif'.format(i))

        # plot1(map_pred, name='{}_HEATMAP_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
        #      show_plot=False, save_output_path=output_folder)

        # plot1(img_sat_, name='{}_ORIG_stepsize{}'.format(f_sat.split('/')[-1][:-4], step_size, nn_output_patch_size[0]),
        #      show_plot=False, save_output_path=output_folder)

    # WRITE GEOTIFF
    geotiff_output_path = os.path.join(output_folder, '{}_HEATMAP.tif'.format(os.path.basename(f_sat)[:-4]))
    write_geotiff(geotiff_output_path, raster_layers=(map_pred*255).astype(np.int16), gdal_ds=f_sat)
    del map_pred
    # WRITE npy (for test)
    # npy_output_path = os.path.join(output_folder, '{}_NUMPY.npy'.format(os.path.basename(f_sat)[:-4]))
    # np.save(npy_output_path, map_pred)

    # from PIL import Image

    # im = Image.fromarray((map_pred * 255).astype(np.uint8))
    # im.save("map_pred_{}.tif".format(i + 1))
