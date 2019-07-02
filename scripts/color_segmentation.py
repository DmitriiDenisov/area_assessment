import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import argparse
import sys
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--nchannels_sat', dest='nchannels_sat', type=int, default=3)
parser.add_argument('--model_name', dest='model_name', type=str, default='color')
parser.add_argument('--density', dest='density', type=float, default=1)
parser.add_argument('--classifier', dest='classifier', type=str, default='sgd')
args = parser.parse_args()

sys.setrecursionlimit(50000)


map_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6_MASK_green.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6_1_MASK_green.tif',
             'Circle_Farms_Sakaka_West.tif', 'Area_Sakaka_Dawmat_Al_Jandal_5-7.tif',
             'Area_Sakaka_Dawmat_Al_Jandal_8-5_mask.tif', 'Area_Sakaka_Dawmat_Al_Jandal_8-6_mask.tif',
             'Area_Sakaka_Dawmat_Al_Jandal_8-7_mask.tif']
sat_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6.tif',
             'Circle_Farms_Sakaka_West.tif', 'Area_Sakaka_Dawmat_Al_Jandal_5-7.tif',
             'Area_Sakaka_Dawmat_Al_Jandal_8-5.tif', 'Area_Sakaka_Dawmat_Al_Jandal_8-6.tif',
             'Area_Sakaka_Dawmat_Al_Jandal_8-7.tif']
train_map_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/map/')
train_sat_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/sat/')

test_sat_folder = os.path.normpath('../sakaka_data/farms/')

output_folder = os.path.normpath('../data/output/')


X = np.zeros((1, args.nchannels_sat), np.uint8)
y = np.array([0], np.uint8)

for i in range(len(map_files)):
    sat = cv2.imread(os.path.join(train_sat_folder, sat_files[i]))
    map = cv2.imread(os.path.join(train_map_folder, map_files[i]), cv2.IMREAD_GRAYSCALE)
    print('Image #{} out of {} is loaded. Sat shape: {}, map shape {}'
          .format(i + 1, len(map_files), sat.shape, map.shape))
    map[map > 0] = 1
    n = sat.shape[0] * sat.shape[1]
    X = np.r_[X, np.zeros((n, args.nchannels_sat), np.uint8)]
    y = np.append(y, np.zeros(n, np.uint8))
    for k in range(args.nchannels_sat):
        X[-n:, k] = sat[:, :, k].flatten().copy()
        y[-n:] = map.flatten().copy()

print('Running color classifier...')
if args.density < 1:
    m = X.shape[0] - 1
    train_obs = np.random.choice(np.arange(m), size=int(args.density * m), replace=False)
    X = X[1:, :][train_obs, :]
    y = y[1:][train_obs]
else:
    X = X[1:, :]
    y = y[1:]

if args.classifier == 'rf':
    model = RandomForestClassifier().fit(X, y)
elif args.classifier == 'lr':
    model = LogisticRegression().fit(X, y)
elif args.classifier == 'sgd':
    model = SGDClassifier(loss='log').fit(X, y)
else:
    print("In color_segmentation: Unknown classifier '{}'".format(args.classifier))
    raise SystemExit

del X; del y
with open(os.path.join(output_folder, args.model_name + '_model'), 'wb') as f:
    pickle.dump(model, f)

print('Predicting train images...')
test_files = [f for f in os.listdir(test_sat_folder) if os.path.isfile(os.path.join(test_sat_folder, f))][2:8]
for i in range(len(sat_files)):
    sat = cv2.imread(os.path.join(train_sat_folder, sat_files[i]))
    print('Image #{} out of {} is loaded. Image shape: {}'.format(i + 1, len(sat_files), sat.shape))
    n = sat.shape[0] * sat.shape[1]
    X = np.zeros((n, args.nchannels_sat), np.uint8)
    for k in range(args.nchannels_sat):
        X[:, k] = sat[:, :, k].flatten().copy()
    y_pred = model.predict_proba(X)[:, 1]
    print('Image is processed')
    cv2.imwrite(os.path.join(output_folder, args.model_name + '_' + sat_files[i] + '.png'),
                y_pred.reshape(sat.shape[0], sat.shape[1]) * 255)
    # np.savetxt(os.path.join(output_folder, args.model_name + '_heatmap{}.csv'.format(i)),
    #            y_pred.reshape(sat.shape[0], sat.shape[1]), '%10.4f', ',')
    print('Predicted heatmap of image #{} is saved'.format(i + 1))

print('Done!')
