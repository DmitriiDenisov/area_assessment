import logging
import numpy as np
import cv2
import os
import sys
# sys.path.append('../area_assesment/images_processing')
# from polygons import mask_to_polygons
from area_assesment.io_operations.data_io import filenames_in_dir
from area_assesment.io_operations.visualization import plot_imgs, plot1

logging.basicConfig(format='%(filename)s:%(lineno)s - %(asctime)s - %(levelname) -8s %(message)s', level=logging.INFO,
                    handlers=[logging.StreamHandler()])


def scoring(mask_true, mask_pred, show_plot=False):
    thresh = 50 / 255 * (mask_pred.max() - mask_pred.min()) + mask_pred.min()
    mask_pred[mask_pred <= thresh] = 0
    mask_pred[mask_pred > 0] = 1

    image, contours, hierarchy = cv2.findContours(((mask_pred == 1) * 255).astype(np.uint8),
                                                  cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    approx_contours = [cv2.approxPolyDP(cnt, 2, True) for cnt in contours if cv2.contourArea(cnt) >= 0.1]
    mask = cv2.drawContours(np.zeros(mask_pred.shape, np.uint8), approx_contours, -1, 1, -1)

    mask_true[mask_true <= 40] = 0
    mask_true[mask_true > 0] = 1

    if show_plot:
        plot_imgs(np.array([[mask_true, mask_pred]]))

    tp, fp, fn = 0, 0, 0
    tp += ((mask == 1) & (mask_true == 1)).sum()
    fp += ((mask == 1) & (mask_true == 0)).sum()
    fn += ((mask == 0) & (mask_true == 1)).sum()
    pre = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * pre * rec / (pre + rec)
    iu = tp / (tp + fp + fn)

    area_pred, area_fact = 0, 0
    area_pred += mask.sum()
    area_fact += mask_true.sum()
    area_error = (area_pred - area_fact) / area_fact

    return {'f1': f1, 'iu': iu, 'area_error': area_error}

dir_mask_pred = os.path.normpath('../sakaka_data/buildings/output/buildings_unet_64x64_epoch407_subpatch32_stepsize32/')
dir_mask_true = os.path.normpath('../sakaka_data/buildings/valid/map/')


masks_true = filenames_in_dir(dir_mask_true, endswith_='tif')
masks_pred = filenames_in_dir(dir_mask_pred, endswith_='npy')

f1, iu, area_error = {}, {}, {}
for i, (f_mask_true, f_mask_pred) in enumerate(list(zip(masks_true, masks_pred))):
    logging.info('f_mask_true:{}, f_mask_pred:{}'.format(f_mask_true, f_mask_pred))
    mask_true = cv2.imread(f_mask_true, cv2.IMREAD_GRAYSCALE)
    mask_pred = np.load(f_mask_pred)
    scores = scoring(mask_true=mask_true, mask_pred=mask_pred, show_plot=False)
    f1[os.path.basename(f_mask_true)[:-4]] = scores['f1']
    iu[os.path.basename(f_mask_true)[:-4]] = scores['iu']
    area_error[os.path.basename(f_mask_true)[:-4]] = scores['area_error']

    logging.info('f1 = {}'.format(round(scores['f1'], 3)))
    logging.info('jaccard(IU) = {}'.format(round(scores['iu'], 3)))
    logging.info('area error = {}'.format(round(scores['area_error'], 3)))

logging.info('Overall images: {}'.format(len(masks_true)))
logging.info('F1 overall: {}'.format(round(sum(f1.values())/len(f1), 3)))
logging.info('IoU (Jaccard) overall: {}'.format(round(sum(iu.values())/len(iu), 3)))
logging.info('Area error overall: {}%'.format(round(sum(area_error.values())/len(area_error), 3)))



