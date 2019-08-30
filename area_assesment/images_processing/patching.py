from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
import logging
import cv2


def array2patches_sklearn(arr, patch_size=(64, 64)):
    """
    Extraction of patches using sklearn function.

    img_sat = plt.imread(image_filepath)
    sat_patches = extract_patches_2d(img_sat, (64, 64), max_patches=1000, random_state=1)

    :param arr: numpy array representation of image with shape (x, y, ...)
    :param patch_size:
    :return: numpy array of extracted patches
    """

    return extract_patches_2d(arr, patch_size)


def array2patches_new(arr, patch_size=(64, 64), step_size=64):
    """
    Extraction of PATCHES FROM IMAGE of shape (x, y, ...) with sliding window.
    Sliding window runs from left upper corner to right bottom corner.
    If step size coincides with patch size height or width there are no overlapping patches.
    Otherwise, there is overlapping, for example horizontal overlap = (nn_input_patch_size[1] - step_size).

    :param arr: numpy array (representation of image) with shape (x, y, ...)
    :param patch_size: tuple (height, width) of sliding window (patch)
    :param step_size: integer, indicates how many pixels are skipped in both the (x, y) direction
    :return: numpy array of shape ((arr.shape[0]-patch.shape[0])//step_size + 1, nn_input_patch_size[0], nn_input_patch_size[1], ...)
    """
    logging.info('ARRAY2PATCHES, arr.shape:{}, patch_size:{}'.format(arr.shape, patch_size))

    # Обрезаем картинку, чтобы подошли размеры под окошко
    slice_0 = (arr.shape[0] - patch_size[0]) // step_size
    slice_1 = (arr.shape[1] - patch_size[1]) // step_size

    arr_cut = arr[0: slice_0 * step_size + patch_size[0],
              0: slice_1 * step_size + patch_size[1]].copy()

    right_range_0 = arr_cut.shape[0] - patch_size[0]
    right_range_1 = arr_cut.shape[1] - patch_size[1]
    if right_range_0 % step_size == 0:
        right_range_0 += 1
    if right_range_1 % step_size == 0:
        right_range_1 += 1

    return np.array([arr_cut[i: i + patch_size[0], j: j + patch_size[1]]
                     for i in range(0, right_range_0, step_size)
                     for j in range(0, right_range_1, step_size)])


def array2patches(arr, patch_size=(64, 64), step_size=64):
    """
    Extraction of PATCHES FROM IMAGE of shape (x, y, ...) with sliding window.
    Sliding window runs from left upper corner to right bottom corner.
    If step size coincides with patch size height or width there are no overlapping patches.
    Otherwise, there is overlapping, for example horizontal overlap = (nn_input_patch_size[1] - step_size).

    :param arr: numpy array (representation of image) with shape (x, y, ...)
    :param patch_size: tuple (height, width) of sliding window (patch)
    :param step_size: integer, indicates how many pixels are skipped in both the (x, y) direction
    :return: numpy array of shape ((arr.shape[0]-patch.shape[0])//step_size + 1, nn_input_patch_size[0], nn_input_patch_size[1], ...)
    """
    logging.info('ARRAY2PATCHES, arr.shape:{}, patch_size:{}'.format(arr.shape, patch_size))
    return np.array([arr[i: i + patch_size[0], j: j + patch_size[1]]
                     for i in range(0, arr.shape[0] - patch_size[0], step_size)
                     for j in range(0, arr.shape[1] - patch_size[1], step_size)])


def patches2array2(patches, img_size, nn_input_patch_size=(64, 64), step_size=16, nn_output_patch_size=(16, 16)):
    """
    Conversion of PATCHES of one image back TO IMAGE WITHOUT OVERLAPPING.
    From every patch the central sub-patch of size step_size (the one which is possible, otherwise some pixels will be
    skipped) is taken. Since, in cases of overlapping the dimension of sub-patch is less than the dimension of patch.
    It is needed to reduce noise on edges of patch.
    Next sub-patches are concatenated into rows and then rows construct output array (image).

    :param patches: numpy array of shape (number of patches, nn_input_patch_size[0], nn_input_patch_size[1])
    :param img_size: tuple (height, width) of image from which patches were generated
    :param patch_size: tuple (height, width) of sliding window (patch)
    :param step_size: should be the same step_size as the step_size during creation of the patches
    :return: numpy array of shape (x, y, ...)
    """

    window_output_size = (step_size, step_size)
    patches_in_row = (img_size[1] - nn_input_patch_size[1]) // step_size + 1
    arr = np.empty((0, patches_in_row * window_output_size[1]))
    for i in range((img_size[0] - nn_input_patch_size[0]) // step_size + 1):
        row = np.concatenate(patches[i * patches_in_row: i * patches_in_row + patches_in_row,
                             nn_input_patch_size[0] // 2 - window_output_size[0] // 2:nn_input_patch_size[0] // 2 +
                                                                                      window_output_size[0] // 2,
                             nn_input_patch_size[1] // 2 - window_output_size[1] // 2:nn_input_patch_size[1] // 2 +
                                                                                      window_output_size[1] // 2],
                             axis=1)
        arr = np.append(arr, row, axis=0)
    return arr


def patches2array_overlap(patches, img_size, patch_size=(64, 64), step_size=64, subpatch_size=(32, 32)):
    """
    Conversion of PATCHES of one image back TO IMAGE WITH OVERLAPPING.
    From every patch the central sub-patch of size nn_output_patch_size is taken and settled into corresponding place into
    layer of shape img_size.
    Next, layers are summarized iteratively - one with the previous one.
    Less step_size increases resolution of the image - more layers influence on one pixel at the output image.

    :param patches: numpy array of shape (number of patches, nn_input_patch_size[0], nn_input_patch_size[1])
    :param img_size: tuple (height, width) of image from which patches were generated
    :param patch_size: tuple (height, width) of sliding window (patch)
    :param step_size: should be the same step_size as the step_size during creation of the patches
    :param subpatch_size: size of subpatch
    :return: numpy array of shape (x, y, ...)
    """
    print('patches2array_overlap: img_size: {}'.format(img_size))
    patches_in_row = (img_size[1] - patch_size[1]) // step_size  # + 1
    patches_in_col = (img_size[0] - patch_size[0]) // step_size  # + 1
    print('patches_in_row: {}, patches_in_col: {}'.format(patches_in_row, patches_in_col))
    arr = np.empty(img_size)
    print('patches2array_overlap: patches.shape: {}'.format(patches.shape))
    print('patches2array_overlap: arr.shape: {}'.format(arr.shape))
    for i in range(patches_in_col):
        print('patches2array_overlap: row {}/{}'.format(i, patches_in_col))
        for j in range(patches_in_row):
            arr2 = np.empty(img_size)
            print(patch_size[0] // 2 - subpatch_size[0] // 2, patch_size[0] // 2 + subpatch_size[0] // 2)
            patch_ij = patches[i * patches_in_col + j,
                       patch_size[0] // 2 - subpatch_size[0] // 2:patch_size[0] // 2 + subpatch_size[0] // 2,
                       patch_size[1] // 2 - subpatch_size[1] // 2:patch_size[1] // 2 + subpatch_size[1] // 2]
            # print('patch_ij.shape: {}'.format(patch_ij.shape))
            arr2[i * step_size:i * step_size + subpatch_size[0],
            j * step_size:j * step_size + subpatch_size[1]] = patch_ij
            # print('patches2array_overlap: {}, {}'.format(i, j))
            arr += arr2
    return arr


def patches2array(patches, img_size, nn_input_patch_size=(64, 64), nn_output_patch_size=(16, 16), step_size=4):
    """
    Conversion of PATCHES of one image back TO IMAGE WITH OVERLAPPING.


    :param patches: numpy array of shape (number of patches, nn_input_patch_size[0], nn_input_patch_size[1])
    :param img_size: tuple (height, width) of image from which patches were generated
    :param nn_input_patch_size: tuple (height, width) of sliding window (patch)
    :param step_size: should be the same step_size as the step_size during creation of the patches
    :param nn_output_patch_size: size of subpatch
    :return: numpy array of shape (x, y, ...)
    """
    logging.info('PATCHES2ARRAY, patches.shape:{}, img_size:{}'.format(patches.shape, img_size))
    logging.debug('patches2array: img_size: {}'.format(img_size))
    patches_in_row = round(
        (img_size[1] - nn_input_patch_size[1]) / step_size)  # (img_size[1] - nn_input_patch_size[1]) // step_size
    patches_in_col = round(
        (img_size[0] - nn_input_patch_size[1]) / step_size)  # (img_size[0] - nn_input_patch_size[0]) // step_size
    logging.debug('patches2array: patches_in_row: {}, patches_in_col: {}'.format(patches_in_row, patches_in_col))
    arr = np.zeros(img_size)
    logging.debug('patches2array: patches.shape: {}'.format(patches.shape))
    logging.debug('patches2array: arr.shape: {}'.format(arr.shape))
    margin_hor = (nn_input_patch_size[0] - nn_output_patch_size[0]) // 2
    margin_vert = (nn_input_patch_size[1] - nn_output_patch_size[1]) // 2
    for i in range(patches_in_col):
        logging.debug('patches2array: row {}/{}'.format(i, patches_in_col))
        for j in range(patches_in_row):
            logging.debug('patches2array: {}, {}'.format(i, j))
            logging.debug('arr_ij: [{}:{}, {}:{}]'.format(
                margin_hor + i * step_size, margin_hor + i * step_size + nn_output_patch_size[0],
                margin_vert + j * step_size, margin_vert + j * step_size + nn_output_patch_size[1]))
            patch_ij = patches[i * patches_in_row + j, :, :]
            logging.debug('patch_ij.shape: {}'.format(patch_ij.shape))
            arr[margin_hor + i * step_size:margin_hor + i * step_size + nn_output_patch_size[0],
            margin_vert + j * step_size:margin_vert + j * step_size + nn_output_patch_size[1]] += patch_ij

    return arr


def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape, flags=cv2.INTER_LINEAR)
    return result
