from itertools import islice
from os import listdir
from os.path import isfile, join
import numpy as np
from keras_preprocessing.image import load_img

from area_assesment.images_processing.patching import array2patches_new
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import just_show_numpy_as_image, plot2


class DataGeneratorCustom:
    def __init__(self, batch_size, train_dir, train_masks_dir, train_nokia_poly, step_size, patch_size, target_channels,
                 nokia_map=False):
        self.batch_size = batch_size
        self.step_size = step_size
        self.patch_size = patch_size
        self.train_dir = train_dir
        self.nokia_map = nokia_map
        if self.nokia_map:
            self.train_nokia_poly = train_nokia_poly
        self.train_masks_dir = train_masks_dir
        self.files_names = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
        self.step_per_epoch = self.get_step_pe_epoch()
        self.target_channels = target_channels

    def __iter__(self):
        while True:
            file_names = np.random.permutation(self.files_names)
            # batch_train_file_names = list(islice(file_names_iter, self.load_one_time_images))
            for f in file_names:
                # print(batch_train_file_names)
                f_image = np.array(load_img(join(self.train_dir, f), color_mode='rgb'))
                # f_image = f_image / 255
                f_mask = np.array(load_img(join(self.train_masks_dir, '{}_MAP.tif'.format(f[:-4])), color_mode='grayscale'))
                f_mask = (f_mask / 255).astype(np.uint8)
                if self.nokia_map:
                    f_nokia = np.array(
                        load_img(join(self.train_nokia_poly, '{}_NOKIA.tif'.format(f[:-4])), color_mode='grayscale'))
                    # f_nokia = self.get_mask_nokia_map(f_nokia) # already done on
                    f_nokia = (f_nokia / 255).astype(np.uint8) # !!!!!!!!
                    f_nokia = f_nokia.reshape((f_nokia.shape + (1,)))
                # x_train_batch = np.array([np.array(load_img(join(self.train_dir, f), color_mode='rgb'))
                #                          / 255 for f in batch_train_file_names])
                # name_map = f[:-4] + '_MAP' + '.tif'
                # y_train_batch = np.array([np.array(load_img(join(self.train_masks_dir, f[:-4] + '_MAP' + '.tif'), color_mode='grayscale'))
                #                          / 255 for f in batch_train_file_names])
                f_mask = f_mask.reshape((f_mask.shape + (1,)))

                # Get pathces
                # aug_x_batch = np.array([], dtype=np.int64).reshape((0, ) + (64, 64, 3))
                # aug_y_batch = np.array([], dtype=np.int64).reshape((0, ) + (64, 64, 1))
                x_train_batch = array2patches_new(f_image, patch_size=self.patch_size, step_size=self.step_size)
                y_train_batch = array2patches_new(f_mask, patch_size=self.patch_size, step_size=self.step_size)
                if self.nokia_map:
                    x_nokia_batch = array2patches_new(f_nokia, patch_size=self.patch_size, step_size=self.step_size)

                # aug_x_batch = np.vstack([aug_x_batch, x_temp])
                # aug_y_batch = np.vstack([aug_y_batch, y_temp])
                del f_image, f_mask
                if self.nokia_map:
                    del f_nokia

                # get random rotations
                x_batch_rotaed = self.get_rotations(x_train_batch, rotation_list=[1, 2, 3])
                y_batch_rotated = self.get_rotations(y_train_batch, rotation_list=[1, 2, 3])
                if self.nokia_map:
                    x_nokia_batch_rotated = self.get_rotations(x_nokia_batch, rotation_list=[1, 2, 3])

                # Unite all of them
                x_train_batch = np.concatenate((x_train_batch, x_batch_rotaed), axis=0)
                y_train_batch = np.concatenate((y_train_batch, y_batch_rotated), axis=0)
                if self.nokia_map:
                    x_nokia_batch = np.concatenate((x_nokia_batch, x_nokia_batch_rotated), axis=0)

                # Append Nokia map polygons to NN
                # if self.nokia_map:
                    # x_train_batch = np.concatenate((x_train_batch, x_nokia_batch), axis=-1)

                # random.permutation
                del x_batch_rotaed, y_batch_rotated
                if self.nokia_map:
                    del x_nokia_batch_rotated

                p = np.random.permutation(len(x_train_batch))
                x_train_batch = x_train_batch[p]
                y_train_batch = y_train_batch[p]
                if self.nokia_map:
                    x_nokia_batch = x_nokia_batch[p]

                # x_train_batch = x_train_batch[:, :, :, 3].reshape(x_train_batch.shape[:-1] + (1, )) # !!!!!!!!!!
                # x_train_batch = x_train_batch[:, :, :, :3] #.reshape(x_train_batch[:, :, :, 3].shape + (1,))  # !!!!!!!!!!
                # yeild
                # iter_over_batch = iter(x_train_batch_new)
                # list(islice(file_names_iter, self.load_one_time_images))
                if self.target_channels == 2:
                    y_train_batch = np.concatenate([y_train_batch, 1 - y_train_batch], axis=-1)
                elif self.target_channels != 1:
                    y_train_batch = y_train_batch.reshape(y_train_batch.shape[:-1])

                for i in range(0, len(x_train_batch), self.batch_size):
                    if self.nokia_map:
                        yield [x_train_batch[i: i + self.batch_size].astype(np.float) / 255, x_nokia_batch[i: i + self.batch_size].astype(np.float)], y_train_batch[i: i + self.batch_size].astype(np.float)
                    else:
                        yield x_train_batch[i: i + self.batch_size].astype(np.float) / 255, y_train_batch[
                                                                                            i: i + self.batch_size].astype(
                            np.float)

    def get_rotations(self, batch, rotation_list):
        target_shape_x = list(batch.shape)
        target_shape_x[0] = 0
        rotated_batch_x = np.array([], dtype=np.uint8).reshape(target_shape_x)
        for i in rotation_list:
            x_temp = np.rot90(batch, axes=(1, 2), k=i)
            rotated_batch_x = np.concatenate((rotated_batch_x, x_temp), axis=0)
        return rotated_batch_x

    def get_step_pe_epoch(self):
        total = 0
        for f in self.files_names:
            temp_img = np.array(load_img(join(self.train_dir, f), color_mode='rgb'))
            num_cols = (temp_img.shape[1] - self.patch_size[1]) // self.step_size
            num_rows = (temp_img.shape[0] - self.patch_size[0]) // self.step_size
            temp = (num_cols + 1) * (num_rows + 1)
            temp = temp * 4
            total += temp
        return total


# train_dir = '../../data/train/big_sat'
##file_names = filenames_in_dir(train_dir, endswith_='.tif')
##train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
##valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]

if __name__ == '__main__':
    # Just checking that wverything works:
    patch_size = (256, 256)
    gen = DataGeneratorCustom(batch_size=4,
                              step_size=64,
                              patch_size=patch_size,
                              target_channels=1,
                              train_dir='../../data/train/sat',
                              train_masks_dir='../../data/train/map',
                              train_nokia_poly='../../data/train/nokia_mask',
                              nokia_map=False
                              )
    i = 0

    # for i in gen:
    #    a, b = i[0], i[1]
    #    pass

    while True:
        i = 0
        for (a, c) in gen:
            just_show_numpy_as_image((a[0] * 255).astype(np.uint8), type='RGB', name='trash/sat_im_{}.tif'.format(i))
            # just_show_numpy_as_image((b[0] * 255).reshape(patch_size), type='not_RGB', name='trash/nokia_map_im_{}.tif'.format(i))
            just_show_numpy_as_image((c[0] * 255).reshape(patch_size), type='not_RGB', name='trash/map_im_{}.tif'.format(i))
            i += 1
            if i > 10:
                1 / 0
            # for j in range(len(a)):
            #    plot2(a[j], b[j].reshape((64, 64)), show_plot=False, save_output_path='', name='test_{}.png'.format(i))
            #    i += 1
            # if i >= 20:
            #    1 / 0
