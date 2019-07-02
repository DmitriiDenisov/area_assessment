import cv2
import os
import numpy as np
from sklearn.linear_model import LinearRegression


dir_input = os.path.normpath('../sakaka_data/stack/input/')
img_names = ['train1', 'train2']
model_names = ['city', 'rural']

x = np.array([], np.float32)
y = np.array([], np.uint8)
files = [f for f in os.listdir(dir_input) if os.path.isfile(os.path.join(dir_input, f))]

for i in range(len(files)):
    print(files[i])
    if (files[i].find('HEATMAP') >= 0):
        x = np.append(x, cv2.imread(os.path.join(dir_input, files[i]), cv2.IMREAD_GRAYSCALE).flatten())
    elif (files[i].find('HEATMAP') < 0) and (files[i].find('_map.') >= 0):
        y = np.append(y, cv2.imread(os.path.join(dir_input, files[i]), cv2.IMREAD_GRAYSCALE).flatten())
        y = np.append(y, cv2.imread(os.path.join(dir_input, files[i]), cv2.IMREAD_GRAYSCALE).flatten())

print(len(x), len(y))