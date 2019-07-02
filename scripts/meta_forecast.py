import cv2
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
import pickle


model_names = ['color_rf10.2', 'cshapes']
# target_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/sat/')
target_folder = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal/')
# output_folder = os.path.normpath('../data/output/')
output_folder = os.path.normpath('../data/output/test/')
model_folder = os.path.normpath('../data/output/')
nchannels_sat = 3

sat_files = [f for f in os.listdir(target_folder) if os.path.isfile(os.path.join(target_folder, f))]
nsats = len(sat_files)
nmodels = len(model_names)


for i in range(nsats):
    print('Image #{}/{}: {}'.format(i + 1, nsats, sat_files[i]))
    sat = cv2.imread(os.path.join(target_folder, sat_files[i]))
    pred = np.zeros((nmodels, sat.shape[0], sat.shape[1]), np.float64)
    for k in range(nmodels):
        print('    Model #{}/{}: {}'.format(k + 1, nmodels, model_names[k]))
        if model_names[k] == 'cshapes':
            retval, image = cv2.threshold(sat, 150, 255, cv2.THRESH_BINARY)
            el0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            image = (255 - cv2.dilate(cv2.dilate(image, el0, iterations=1), el, iterations=1)).astype(np.float64).sum(2)
            image /= image.max()
            pred[k, :, :] = image.copy()
        else:
            with open(os.path.join(model_folder, model_names[k] + '_model'), 'rb') as f:
                model = pickle.load(f)
            X = np.zeros((sat.shape[0] * sat.shape[1], nchannels_sat), np.uint8)
            for j in range(nchannels_sat):
                X[:, j] = sat[:, :, j].flatten().copy()
            pred[k, :, :] = model.predict_proba(X)[:, 1].reshape(sat.shape[0], sat.shape[1])
    cv2.imwrite(os.path.join(output_folder, 'mean_' + sat_files[i] + '.png'), pred.mean(0) * 255)

