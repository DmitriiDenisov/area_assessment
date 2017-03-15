import os
import cv2
import matplotlib.pyplot as plt


def filenames_in_dir(dir_, endswith_):
    return [os.path.join(dir_, filename) for filename in os.listdir(dir_) if filename.endswith(endswith_)]


def read_img(filename_):
    """ """
    img = cv2.imread(filename_)
    return img
