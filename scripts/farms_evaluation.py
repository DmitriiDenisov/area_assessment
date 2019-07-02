import cv2
import numpy as np
import os


map_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6_MASK.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6_MASK.tif',
             'Circle_Farms_Sakaka_West.tif']
sat_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6.tif',
             'Circle_Farms_Sakaka_West.tif']
train_map_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/map/')
train_sat_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/sat/')
output_folder = os.path.normpath('../data/output/')

model_names = ['color_rf10', 'color_rf20', 'color_sgd', 'cshapes', 'meta']
nmodels = len(model_names)


thresh_range = np.arange(0.05, 1., 0.05)
nthresh = len(thresh_range)
for i in range(len(map_files)):
    map = cv2.imread(os.path.join(train_map_folder, map_files[i]), cv2.IMREAD_GRAYSCALE)
    map[map > 0] = 1
    for k in range(nmodels):
        heat = cv2.imread(os.path.join(output_folder, model_names[k] + '_' + sat_files[i] + '.png'),
                          cv2.IMREAD_GRAYSCALE).astype(np.float64) / 255
        pred = 1 * (heat >= .5)
        tp = ((pred == 1) & (map == 1)).sum()
        tn = ((pred == 1) & (map == 0)).sum()
        fn = ((pred == 0) & (map == 1)).sum()
        pre = tp / (tp + tn)
        rec = tp / (tp + fn)
        f1 = 2 * pre * rec / (pre + rec)
        ja = tp / (tp + tn + fn)
        ar = pred.sum() / map.sum() - 1
        cv2.imwrite(os.path.join(output_folder, 'eval' + model_names[k] + '_' + sat_files[i] + '.png'), 255 * pred)
        print('Image {}, model {}: f1 = {}, jacard = {}, area error = {}'
              .format(sat_files[i], model_names[k], round(f1, 3), round(ja, 3), round(ar, 3)))

