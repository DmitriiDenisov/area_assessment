from itertools import islice
from os import listdir
from os.path import isfile, join
import numpy as np
from keras_preprocessing.image import load_img

# numpy.rot90(orignumpyarray,3)
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import just_show_numpy_as_image


class DataGeneratorCustom:
    def __init__(self, batch_size, train_dir, train_masks_dir, load_one_time_images=3):
        self.batch_size = batch_size
        self.train_dir = train_dir
        self.train_masks_dir = train_masks_dir
        self.files_names = [f for f in listdir(train_dir) if isfile(join(train_dir, f))]
        self.load_one_time_images = load_one_time_images

    def get_rotations(self, x_train_batch, y_train_batch, rotation_list):
        target_shape_x = list(x_train_batch.shape)
        target_shape_x[0] = 0
        target_shape_y = list(y_train_batch.shape)
        target_shape_y[0] = 0
        rotated_batch_x = np.array([]).reshape(target_shape_x)
        rotated_batch_y = np.array([]).reshape(target_shape_y)
        for i in rotation_list:
            x_temp = np.rot90(x_train_batch, axes=(1, 2), k=i)
            y_temp = np.rot90(y_train_batch, axes=(1, 2), k=i)
            rotated_batch_x = np.concatenate((rotated_batch_x, x_temp), axis=0)
            rotated_batch_y = np.concatenate((rotated_batch_y, y_temp), axis=0)
        return rotated_batch_x, rotated_batch_y

    def __iter__(self):
        while True:
            file_names = np.random.permutation(self.files_names)
            file_names_iter = iter(file_names)
            batch_train_file_names = list(islice(file_names_iter, self.load_one_time_images))
            while batch_train_file_names:
                # print(batch_train_file_names)
                x_train_batch = np.array([np.array(load_img(join(self.train_dir, f), grayscale=False))
                                          / 255 for f in batch_train_file_names])
                y_train_batch = np.array([np.array(load_img(join(self.train_masks_dir, f), grayscale=True))
                                          / 255 for f in batch_train_file_names])
                y_train_batch = y_train_batch.reshape((y_train_batch.shape + (1,)))

                # get random rotations
                x_batch_rotaed, y_batch_rotated = self.get_rotations(x_train_batch, y_train_batch, [1, 2, 3])
                # Unite all of them
                x_train_batch_new = np.concatenate((x_train_batch, x_batch_rotaed), axis=0)
                y_train_batch_new = np.concatenate((y_train_batch, y_batch_rotated), axis=0)
                # random.permutation
                p = np.random.permutation(len(x_train_batch_new))
                x_train_batch_new = x_train_batch_new[p]
                y_train_batch_new = y_train_batch_new[p]

                # yeild
                # iter_over_batch = iter(x_train_batch_new)
                # list(islice(file_names_iter, self.load_one_time_images))

                for i in range(0, len(x_train_batch_new), self.batch_size):
                    yield x_train_batch_new[i: i + self.batch_size], y_train_batch_new[i: i + self.batch_size]
                batch_train_file_names = list(islice(file_names_iter, self.load_one_time_images))


# Wrapper over reading-files generator
def create_aug_gen(in_gen):
    for in_x, in_y in in_gen:
        g_x = image_gen.flow(255 * in_x, in_y,
                             batch_size=in_x.shape[0])
        x, y = next(g_x)
        yield x / 255.0, y


train_dir = '../../data/test_data/sat'
file_names = filenames_in_dir(train_dir, endswith_='.tif')
train_file_names = np.random.permutation(file_names)[:round(0.8 * len(file_names))]
valid_file_names = np.random.permutation(file_names)[round(0.8 * len(file_names)):]
gen = iter(
    DataGeneratorCustom(batch_size=2, train_dir='../../data/test_data/sat', train_masks_dir='../../data/test_data/map'))
next(gen)
