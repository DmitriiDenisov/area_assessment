import cv2
import numpy as np
import matplotlib
# from area_assesment.images_processing.polygons import mask_to_polygons
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from area_assesment.images_processing.patching import array2patches, patches2array_overlap, patches2array
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_img_mask_pred, plot_img_mask, plot_img
from area_assesment.neural_networks.cnn import cnn_v1, cnn_v3

# MODEL
model = cnn_v3()
# model.summary()
model.load_weights('weights/sakaka_cnn_v3_w_11.h5')
# model.load_weights('weights/cnn_v3_w_63.h5')


# PATCHING SETTINGS
nn_input_patch_size = (64, 64)  # how to patching validation image: size of input layer of MODEL
step_size = 16
nn_output_patch_size = (16, 16)  # size of central sub-patch of predicted patch, to transform predicted patches into image


# VALIDATION ON ALL IMAGES IN THE VALID DIRECTORY
# dir_valid = '../../data/mass_buildings/valid/'  # '../../sakaka_data/train/'  # '../../data/mass_buildings/valid/'
dir_valid = '../../sakaka_data/train/'
dir_valid_sat = dir_valid + 'sat/'
dir_valid_map = dir_valid + 'map/'
valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tif')[:]
valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')[:]
for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))):
    print('VALID IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
    img_sat_, img_map_ = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map_[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat_.astype('float32')
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

    # PLOT IMG, MASK_TRUE, MASK_PRED
    plot_img_mask_pred(img_sat, img_map, map_pred,
                       name='VALID_IMG{}_cnn_v3_stepsize{}'.format(i + 1, step_size),
                       show_plot=True, save_output_path='../../plots/sakaka4/')

    # PLOT OVERLAY IMG, MASK_PRED
    # margin_hor = (nn_input_patch_size[0] - nn_output_patch_size[0]) // 2
    # margin_vert = (nn_input_patch_size[1] - nn_output_patch_size[1]) // 2
    # plot_img_mask(img_sat[margin_hor: -margin_hor, margin_vert: -margin_vert],
    #               map_pred[margin_hor: -margin_hor, margin_vert: -margin_vert],
    #               name='VALID_IMG{}_OVERLAY_cnn_v3_stepsize{}'.format(i + 1, step_size),
    #               overlay=True, alpha=0.3, show_plot=True, save_output_path='../../plots/sakaka4/')

    # plot_img(img_sat[margin_hor: -margin_hor, margin_vert: -margin_vert],
    #          name='VALID_IMG{}_SAT_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path='../../plots/sakaka4/')
    # plot_img(img_map[margin_hor: -margin_hor, margin_vert: -margin_vert],
    #          name='VALID_IMG{}_MAP_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path='../../plots/sakaka4/')
    # plot_img(map_pred[margin_hor: -margin_hor, margin_vert: -margin_vert],
    #          name='VALID_IMG{}_PRED_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #          show_plot=False, save_output_path='../../plots/sakaka4/')

    # np.save('VALID_IMG{}_SAT_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #         img_sat_[margin_hor: -margin_hor, margin_vert: -margin_vert])
    # np.save('VALID_IMG{}_MAP_stepsize{}_subpatchsize{}'.format(i + 1, step_size, nn_output_patch_size[0]),
    #         img_map[margin_hor: -margin_hor, margin_vert: -margin_vert])
    # np.save('VALID_IMG{}_PRED_stepsize{}_subpatchsize{}.npy'.format(i + 1, step_size, nn_output_patch_size[0]),
    #         map_pred[margin_hor: -margin_hor, margin_vert: -margin_vert])

    # SCORE EVALUATION
    # score = model.evaluate(sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    ## print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:map_pred.shape[0], :map_pred.shape[1]].reshape(-1), map_pred.reshape(-1))))

