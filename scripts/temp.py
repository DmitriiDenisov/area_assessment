from area_assesment.images_processing.patching import *
from area_assesment.io_operations.data_io import *
from area_assesment.io_operations.visualize_img import *
from area_assesment.neural_networks.cnn import cnn_v1

dir_train = '../../data/train/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'

train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tiff')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')
print(train_sat_files)
print(train_map_files)

for (f_sat, f_map) in list(zip(train_sat_files, train_map_files)):
    img_sat, img_map = read_img(f_sat), read_img(f_map)
    # print(img_map.shape)
    # plot_img_mask(img_sat, img_map)

    img_sat_patches = extract_patches_2d(img_sat, (64, 64), max_patches=10, random_state=1)
    img_map_patches = extract_patches_2d(img_map[:, :, 2], (64, 64), max_patches=10, random_state=1)
    # img_sat_map_patches = list(zip(img_sat_patches, img_map_patches))
    # print(len(img_sat_map_patches))

    model = cnn_v1()
    model.fit(img_sat_patches, img_map_patches, epochs=5)

    for i in range(10):
        print(img_sat_patches[i].shape, img_map_patches[i].shape)
        map_patch_pred = model.predict(img_sat_patches[i].reshape(1, 64, 64, 3)).reshape(64, 64)
        # print(map_patch_pred.shape)
        # print(map_patch_pred)
        plot_img_mask_pred(img_sat_patches[i], img_map_patches[i], map_patch_pred)
