import cv2
import numpy as np
import matplotlib.pyplot as plt
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import *


if __name__ == '__main__':
    dir_ = '../../sakaka_data/output/sakaka_grabcut/'
    dir_sat = dir_ + 'sat/'
    dir_map = dir_ + 'map/'
    dir_pred = dir_ + 'pred/'
    sat_files = filenames_in_dir(dir_sat, endswith_='.tif')
    map_files = filenames_in_dir(dir_map, endswith_='.tif')
    pred_files = filenames_in_dir(dir_pred, endswith_='.npy')

    for i, (sat, map, pred) in enumerate(list(zip(sat_files, map_files, pred_files))):
        print(i+1, sat, map, pred)
        sat = cv2.imread(sat)
        map = cv2.imread(map)[:, :, 0]
        # sat, map, pred = np.load(sat), np.load(map), np.load(pred)
        pred = np.load(pred)
        print(sat.shape, map.shape, pred.shape)
        plot_img_mask_pred(sat, map, pred)

        mask = np.zeros(sat.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        mask[pred < np.percentile(pred, 95)] = 0
        mask[pred >= np.percentile(pred, 95)] = 1
        cv2.grabCut(sat, mask, None, bgdModel, fgdModel, 100, cv2.GC_INIT_WITH_MASK)
        mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        sat = sat * mask[:, :, np.newaxis]
        plot_img_mask(sat, map, overlay=True, alpha=0.5)
        plot_img_mask(sat, pred, overlay=True, alpha=0.5)


