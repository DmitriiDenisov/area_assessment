import os
import numpy as np
import cv2


def equalizeHist_rgb(img):
    # return rgb/255.0
    norm = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    norm[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    norm[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    norm[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return norm

