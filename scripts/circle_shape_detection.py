import cv2
import numpy as np
import os


nchannels_sat = 3
model_name = 'cshapes'
density = 1

map_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6_MASK.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6_MASK.tif',
             'Circle_Farms_Sakaka_West.tif', 'Area_Sakaka_Dawmat_Al_Jandal_5-7.tif']
sat_files = ['Area_Sakaka_Dawmat_Al_Jandal_1-6.tif', 'Area_Sakaka_Dawmat_Al_Jandal_2-6.tif',
             'Circle_Farms_Sakaka_West.tif', 'Area_Sakaka_Dawmat_Al_Jandal_5-7.tif']
train_map_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/map/')
train_sat_folder = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/circle_farms/train/sat/')

test_sat_folder = os.path.normpath('../sakaka_data/farms/')

output_folder = os.path.normpath('../data/output/')


for i in range(len(sat_files)):
    original = cv2.imread(os.path.join(train_sat_folder, sat_files[i]), cv2.IMREAD_GRAYSCALE)
    retval, image = cv2.threshold(original, 150, 255, cv2.THRESH_BINARY)
    el0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    el = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    image = 255 - cv2.dilate(image, el, iterations=1)
    cv2.imwrite(os.path.join(output_folder, model_name + '_' + sat_files[i] + '.png'), image)
    # np.savetxt(os.path.join(output_folder, model_name + '_' + sat_files[i] + '.csv'), image.astype(np.float64) / 255,
    #            '%10.4f', ',')
