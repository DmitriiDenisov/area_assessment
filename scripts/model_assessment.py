import numpy as np
import cv2
import os
import sys
# sys.path.append('../area_assesment/images_processing')
# from polygons import mask_to_polygons
from area_assesment.io_operations.visualization import plot_imgs, plot1


def scoring(mask_true, mask_pred):
    thresh = 50 / 255 * (mask_pred.max() - mask_pred.min()) + mask_pred.min()
    mask_pred[mask_pred <= thresh] = 0
    mask_pred[mask_pred > 0] = 1

    image, contours, hierarchy = cv2.findContours(((mask_pred == 1) * 255).astype(np.uint8),
                                                  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    approx_contours = [cv2.approxPolyDP(cnt, 2, True) for cnt in contours if cv2.contourArea(cnt) >= 0.1]
    mask = cv2.drawContours(np.zeros(mask_pred.shape, np.uint8), approx_contours, -1, 1, -1)

    mask_true[mask_true <= 40] = 0
    mask_true[mask_true > 0] = 1

    # plot_imgs(np.array([[mask_true, mask_pred]]))
    # plot1(mask_true, show_plot=True)
    # plot1(mask, show_plot=True)

    tp = fp = fn = 0
    tp += ((mask == 1) & (mask_true == 1)).sum()
    fp += ((mask == 1) & (mask_true == 0)).sum()
    fn += ((mask == 0) & (mask_true == 1)).sum()

    area_pred = area_fact = 0
    area_pred += mask.sum()
    area_fact += mask_true.sum()

    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)

    print('f1 = {}'.format(round(f1, 3)))
    print('jaccard(IU) = {}'.format(round(tp / (tp + fp + fn), 3)))
    print('area error = {}'.format(round((area_pred - area_fact) / area_fact, 3)))


mask_pred = np.load('../sakaka_data/buildings/output/buildings_unet_64x64_epoch407_subpatch32_stepsize32/Dawmat_Al_Jandal_04_11_HEATMAP.npy')
mask_true = cv2.imread('../sakaka_data/buildings/valid/map/Dawmat_Al_Jandal_04_11.tif', cv2.IMREAD_GRAYSCALE)

scoring(mask_true=mask_true, mask_pred=mask_pred)



