import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import argparse
import sys
import pickle

sys.setrecursionlimit(50000)


nchannels_sat = 3

map_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6_MASK.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6_MASK.tif',
             'Circle_Farms_Sakaka_West.tif']
sat_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6.tif',
             'Circle_Farms_Sakaka_West.tif']
train_map_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/map/')
train_sat_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/sat/')

test_sat_folder = os.path.normpath('../sakaka_data/farms/')

output_folder = os.path.normpath('../data/output/')

model_names = ['color_rf10', 'color_rf20', 'color_sgd', 'cshapes']
nmodels = len(model_names)


models = list(range(nmodels))
for i in range(nmodels):
    if model_names[i] != 'cshapes':
        with open(os.path.join(output_folder, model_names[i] + '_model'), 'rb') as f:
            models[i] = pickle.load(f)

print('Loading meta data...')
X = np.zeros((1, nmodels), np.uint8)
y = np.array([0], np.uint8)
for k in range(len(map_files)):
    map = cv2.imread(os.path.join(train_map_folder, map_files[k]), cv2.IMREAD_GRAYSCALE)
    map[map > 0] = 1
    n = map.shape[0] * map.shape[1]
    X = np.r_[X, np.zeros((n, nmodels), np.uint8)]
    y = np.append(y, map.flatten().copy())
    for i in range(nmodels):
        img = cv2.imread(os.path.join(output_folder, model_names[i] + '_' + sat_files[k] + '.png'),
                         cv2.IMREAD_GRAYSCALE)
        X[-n:, i] = img.flatten().copy()

X = X[1:, :]
y = y[1:]

print('Running meta classifier...')
density = 0.2
m = X.shape[0]
train_obs = np.random.choice(np.arange(m), size=int(density * m), replace=False)
meta = RandomForestClassifier().fit(X[train_obs, :], y[train_obs])
del X; del y

print('Predicting train images...')
for i in range(len(sat_files)):
    sat = cv2.imread(os.path.join(train_sat_folder, sat_files[i]))
    n = sat.shape[0] * sat.shape[1]
    X = np.zeros((n, nmodels), np.uint8)
    for k in range(nmodels):
        X[:, k] = cv2.imread(os.path.join(output_folder, model_names[i] + '_' + sat_files[i] + '.png'),
                             cv2.IMREAD_GRAYSCALE).flatten()
    print('Heatmaps of image #{} out of {} are loaded'.format(i + 1, len(sat_files)))
    y_pred = meta.predict_proba(X)[:, 1]
    print('Image is processed')
    cv2.imwrite(os.path.join(output_folder, 'meta_' + sat_files[i] + '.png'),
                y_pred.reshape(sat.shape[0], sat.shape[1]) * 255)
    print('Predicted train heatmap of image #{} is saved'.format(i + 1))

print('Predicting test images...')
test_files = [f for f in os.listdir(test_sat_folder) if os.path.isfile(os.path.join(test_sat_folder, f))]
for i in range(len(test_files)):
    sat = cv2.imread(os.path.join(test_sat_folder, test_files[i]))
    print('Image #{} out of {} is loaded. Image shape: {}'.format(i + 1, len(test_files), sat.shape))
    n = sat.shape[0] * sat.shape[1]
    X = np.zeros((n, nmodels), np.uint8)
    for k in range(nmodels):
        if model_names[k] == 'cshapes':
            retval, image = cv2.threshold(sat, 125, 255, cv2.THRESH_BINARY)
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            image = (255 - cv2.dilate(image, el, iterations=2)).astype(np.float64).sum(2)
            image[image > 0] = 1
            X[:, k] = image.flatten().copy()
        else:
            Z = np.zeros((n, nchannels_sat), np.uint8)
            for j in range(nchannels_sat):
                Z[:, j] = sat[:, :, j].flatten().copy()
            X[:, k] = models[k].predict_proba(Z)[:, 1].flatten()
        print('Partial algorithm #{} out of {} is completed'.format(k + 1, nmodels))
    y_pred = meta.predict_proba(X)[:, 1]
    print('Image is processed')
    cv2.imwrite(os.path.join(output_folder, 'meta_' + test_files[i] + '.png'),
                y_pred.reshape(sat.shape[0], sat.shape[1]) * 255)
    print('Predicted test heatmap of image #{} is saved'.format(i + 1))
