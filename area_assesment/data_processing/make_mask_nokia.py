import os
from os.path import join

import numpy as np
from PIL import Image
from keras_preprocessing.image import load_img


def get_mask_nokia_map(f_nokia):
    pixel1 = [238, 243, 245]
    mask = (f_nokia[:, :, ] == pixel1).all(axis=-1).astype(np.uint8)
    return mask * 255


source_dir = '../../data/train/nokia_map'
target_dir = '../../data/train/nokia_mask'
list_of_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f[-4:] == '.tif']

for f in list_of_files:
    print('Processing for file:{}'.format(f))
    f_nokia = np.array(load_img(join(source_dir, f), grayscale=False))
    mask = get_mask_nokia_map(f_nokia)
    im = Image.fromarray(mask)
    path_ = os.path.join(target_dir, "{}_NOKIA.tif".format(f[:-4]))
    im.save(path_)
