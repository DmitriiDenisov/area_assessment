# from area_assesment.images_processing.patching import *
# from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

from area_assesment.io_operations.data_io import filenames_in_dir

dir_train = '../../data/mass_buildings/train/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'

train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tiff')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')

for (f_sat, f_map) in list(zip(train_sat_files, train_map_files)):
    print(f_sat, f_map)
    img_sat, img_map = plt.imread(f_sat), plt.imread(f_map)
    img_map = img_map[:, :, 0] / 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255
    print(type(img_sat))
    print(img_sat.shape)
    print(img_sat.max())
    # ret, img_map = cv2.threshold(img_map[:, :, 2], 127, 255, cv2.THRESH_BINARY)
    # img_map = img_map[:, :, 2]/max(img_map[:, :, 2])

    # # plot_img_mask(img_sat, img_map)
    #
    # img_sat_patches = extract_patches_2d(img_sat, (64, 64), max_patches=1000, random_state=1)
    # img_map_patches = extract_patches_2d(img_map, (64, 64), max_patches=1000, random_state=1)
    #
    # model = baseline_model()
    # model.fit(img_sat_patches, img_map_patches, epochs=2)
    # model.save_weights('weights/w1.h5')
    # # model.load_weights('weights/w1.h5')
    #
    # for i in range(5):
    #     print(img_sat_patches[i].shape, img_map_patches[i].shape)
    #     map_patch_pred = model.predict(img_sat_patches[i].reshape(1, 64, 64, 3)).reshape(64, 64)
    #     plot_img_mask_pred(img_sat_patches[i], img_map_patches[i], map_patch_pred)