import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from sklearn.feature_extraction.image import *
from keras.models import Sequential
from keras.layers import *
from sklearn.metrics import jaccard_similarity_score


def filenames_in_dir(dir_, endswith_):
    return [os.path.join(dir_, filename) for filename in os.listdir(dir_) if filename.endswith(endswith_)]


def array2patches(arr, patch_size=(64, 64), step_size=64):
    return np.array([arr[i: i + patch_size[0], j: j + patch_size[1]]
                     for i in range(0, arr.shape[0] - patch_size[0], step_size)
                     for j in range(0, arr.shape[1] - patch_size[1], step_size)])


def patches2array(patches, img_size, patch_size=(64, 64)):
    patches_in_row = img_size[1] // patch_size[1]
    arr = np.empty((0, patches_in_row * patch_size[1]))
    # print('arr', arr.shape)
    for i in range(img_size[0] // patch_size[0]):
        row = np.concatenate(patches[i * patches_in_row: i * patches_in_row + patches_in_row], axis=1)
        arr = np.append(arr, row, axis=0)
    return arr


def patches2array2(patches, img_size, patch_size=(64, 64), step_size=64):
    window_output_size = (step_size, step_size)
    patches_in_row = (img_size[1] - patch_size[1]) // step_size + 1
    arr = np.empty((0, patches_in_row * window_output_size[1]))
    print('patches.shape: {}'.format(patches.shape))
    print('arr.shape: {}'.format(arr.shape))
    for i in range((img_size[0] - patch_size[0]) // step_size + 1):
        print('1 patch.shape: {}'.format(patches[i * patches_in_row: (i + 1) * patches_in_row].shape))
        print('2 patch.shape: {}'.format(patches[i * patches_in_row: i * patches_in_row + patches_in_row,
                             patch_size[0]//2 - window_output_size[0]//2:patch_size[0]//2 + window_output_size[0]//2,
                             patch_size[1] // 2 - window_output_size[1]//2:patch_size[1]//2 + window_output_size[1]//2].shape))

        row = np.concatenate(patches[i * patches_in_row: i * patches_in_row + patches_in_row,
                             patch_size[0]//2 - window_output_size[0]//2:patch_size[0]//2 + window_output_size[0]//2,
                             patch_size[1] // 2 - window_output_size[1]//2:patch_size[1]//2 + window_output_size[1]//2],
                             axis=1)
        arr = np.append(arr, row, axis=0)
    return arr


def patches2array3(patches, img_size, patch_size=(64, 64), step_size=64, patch_output_size=()):
    # subpatch_size = (step_size, step_size)
    print('img_size: {}'.format(img_size))
    patches_in_row = (img_size[1] - patch_size[1]) // step_size
    patches_in_col = (img_size[0] - patch_size[0]) // step_size
    # arr = np.empty((patches_in_col * subpatch_size[0], patches_in_row * subpatch_size[1]))
    arr = np.empty(img_size)
    print('patches.shape: {}'.format(patches.shape))
    print('arr.shape: {}'.format(arr.shape))
    for i in range(patches_in_col):
        for j in range(patches_in_row):
            arr2 = np.empty(img_size)
            patch_ij = patches[i * patches_in_row + j,
                       patch_size[0]//2 - patch_output_size[0]//2:patch_size[0]//2 + patch_output_size[0]//2,
                       patch_size[1]//2 - patch_output_size[1]//2:patch_size[1]//2 + patch_output_size[1]//2]
            arr2[i*step_size:i*step_size+patch_output_size[0], j*step_size:j*step_size+patch_output_size[1]] = patch_ij
            print('{}, {}'.format(i, j))
            arr += arr2

    return arr


def cnn_v1():
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=16, strides=4, activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=2, strides=1))
    model.add(Conv2D(filters=112, kernel_size=4, strides=1, activation='relu'))
    model.add(Conv2D(filters=80, kernel_size=3, strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(64 ** 2, activation='sigmoid'))
    model.add(Reshape((64, 64)))
    # print(model.output_shape)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def plot_img_mask(img, mask):
    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(10, 10))
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask), ax[1].set_title('mask_true'), ax[1].axis('off')
    plt.tight_layout(), plt.show()


def plot_img_mask_pred(img, mask_true, mask_pred, name=None, show_plot=True, save_output_path=None):
    fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(20, 10))
    ax[0].imshow(img), ax[0].set_title('image'), ax[0].axis('off')
    ax[1].imshow(mask_true), ax[1].set_title('mask_true'), ax[1].axis('off')
    ax[2].imshow(mask_pred), ax[2].set_title('mask_pred'), ax[2].axis('off')
    plt.tight_layout()
    if name:
        plt.suptitle(name)
    if show_plot:
        plt.show()
    if save_output_path:
        fig.savefig(save_output_path + name + '.pdf', dpi=500)


# TRAIN, CREATE PATCHES FROM ALL IMAGES
dir_train = '../../data/mass_buildings/train/'
dir_train_sat = dir_train + 'sat/'
dir_train_map = dir_train + 'map/'
train_sat_files = filenames_in_dir(dir_train_sat, endswith_='.tiff')
train_map_files = filenames_in_dir(dir_train_map, endswith_='.tif')

sat_patches, map_patches = np.empty((0, 64, 64, 3)), np.empty((0, 64, 64))
for i, (f_sat, f_map) in enumerate(list(zip(train_sat_files[:1], train_map_files[:1]))):
    print('TRAIN, PATCHING IMG: {}/{}, {}, {}'.format(i + 1, len(train_sat_files), f_sat, f_map))
    img_sat, img_map = cv2.imread(f_sat), cv2.imread(f_map)
    img_map = img_map[:, :, 2].astype('float32')
    img_map /= 255
    img_sat = img_sat.astype('float32')
    img_sat /= 255

    print('img_sat.shape: {}'.format(img_sat.shape))
    # img_sat_patches = extract_patches_2d(img_sat, (64, 64), max_patches=1000, random_state=1)
    # img_map_patches = extract_patches_2d(img_map, (64, 64), max_patches=1000, random_state=1)
    img_sat_patches, img_map_patches = array2patches(img_sat, step_size=32), array2patches(img_map, step_size=32)
    print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
    print('img_map_patches.shape: {}'.format(img_map_patches.shape))

    sat_patches, map_patches = np.append(sat_patches, img_sat_patches, 0), np.append(map_patches, img_map_patches, 0)

print('sat_patches.shape: {}'.format(sat_patches.shape))
print('map_patches.shape: {}'.format(map_patches.shape))

# for (sat_patch, map_patch) in list(zip(sat_patches, map_patches)):
#     plot_img_mask(sat_patch, map_patch)


# MODEL
model = cnn_v1()
# model.summary()
model.load_weights('weights/w_33.h5')
tb_callback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
model.fit(sat_patches, map_patches, epochs=20, callbacks=[tb_callback])
model.save_weights('weights/w_{}.h5'.format(34))


# VALIDATION
# dir_valid = '../../data/mass_buildings/valid/'
# dir_valid_sat = dir_valid + 'sat/'
# dir_valid_map = dir_valid + 'map/'
# valid_sat_files = filenames_in_dir(dir_valid_sat, endswith_='.tiff')
# valid_map_files = filenames_in_dir(dir_valid_map, endswith_='.tif')
# for i, (f_sat, f_map) in enumerate(list(zip(valid_sat_files, valid_map_files))):
#     print('VALID IMG: {}/{}, {}, {}'.format(i + 1, len(valid_sat_files), f_sat, f_map))
#     img_sat, img_map = cv2.imread(f_sat)[:600, :600], cv2.imread(f_map)[:600, :600]
#     img_map = img_map[:, :, 2].astype('float32')
#     img_map /= 255
#     img_sat = img_sat.astype('float32')
#     img_sat /= 255
#
#     step_size = 2
#
#     img_sat_patches = array2patches(img_sat, step_size=step_size)
#     print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
#
#     map_patches_pred = model.predict(img_sat_patches)
#     print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))
#
#     # VALIDATION ON ONE PATCH
#     # for i in range(img_sat_patches.shape[0]):
#     #     print(img_sat_patches[i].shape, img_map_patches[i].shape)
#     #     map_patch_pred = model.predict(img_sat_patches[i].reshape(1, 64, 64, 3)).reshape(64, 64)
#     #     print(map_patch_pred)
#     #     plot_img_mask_pred(img_sat_patches[i], img_map_patches[i], map_patch_pred)
#
#     subpatch_size = (8, 8)
#     img_map_pred = patches2array3(map_patches_pred, img_size=img_sat.shape[:2], step_size=step_size,
#                                   subpatch_size=subpatch_size)
#     print('img_map_pred.shape: {}'.format(img_map_pred.shape))
#
#     plot_img_mask_pred(img_sat, img_map, img_map_pred,
#                        name='3_VALID_IMG{}_stepsize{}_patchoutputsize{}'.format(i+1, step_size, subpatch_size[0]),
#                        show_plot=False, save_output_path='../../plots/')

    # score = model.evaluate(img_sat_patches, array2patches(img_map))
    # print('Score: {}'.format(score))
    # print('Jaccard:{}'.format(jaccard_similarity_score(img_map[:img_map_pred.shape[0], :img_map_pred.shape[1]].reshape(-1), img_map_pred.reshape(-1))))


# TEST
# dir_test = '../../data/mass_buildings/test/'
# dir_test_sat = dir_test + 'sat/'
# test_sat_files = filenames_in_dir(dir_test_sat, endswith_='.tiff')
# for i, f_sat in enumerate(test_sat_files):
#     print('TEST IMG: {} ({}), {}'.format(i+1, len(test_sat_files), f_sat))
#     # img_sat = plt.imread(f_sat)
#     img_sat = plt.imread(f_sat)
#     img_sat = img_sat.astype('float32')
#     img_sat /= 255
#     print('img_sat.shape: {}'.format(img_sat.shape))
#
#     img_sat_patches = array2patches(img_sat)
#     print('img_sat_patches.shape: {}'.format(img_sat_patches.shape))
#
#     map_patches_pred = model.predict(img_sat_patches)
#     print('map_patches_pred.shape: {}'.format(map_patches_pred.shape))
#     img_map_pred = patches2array(map_patches_pred, img_size=img_sat.shape[:2])
#     print('img_map_pred.shape: {}'.format(img_map_pred.shape))
#
#     plot_img_mask(img_sat, img_map_pred)
