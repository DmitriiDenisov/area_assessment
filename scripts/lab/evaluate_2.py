import numpy as np
import cv2
import os
import sys
sys.path.append('../area_assesment/images_processing')
from polygons import mask_to_polygons


np.random.seed(111)
train_nums = np.random.choice(np.arange(112), size=int(0.7 * 112), replace=False)
valid_nums = np.delete(np.arange(112), train_nums)

folder_heatmap = '../sakaka_data/buildings/output/'
heats = [f for f in os.listdir(folder_heatmap) if os.path.isfile(os.path.join(folder_heatmap, f))]
folder_mask = os.path.normpath('/storage/_pdata/sakaka/sakaka_data/buildings/test_evaluate/map')
maps = sorted([f for f in os.listdir(folder_mask) if os.path.isfile(os.path.join(folder_mask, f))])
# maps = [maps[i] for i in valid_nums]
folder_poly = os.path.normpath('../sakaka_data/buildings/output/u1/')

tp = fp = fn = 0
area_pred = area_fact = 0

map = np.loadtxt('pred0', delimiter=',')
thresh = 50 / 255 * (map.max() - map.min()) + map.min()
map[map <= thresh] = 0; map[map > 0] = 1
image, contours, hierarchy = cv2.findContours(((map == 1) * 255).astype(np.uint8),
                                              cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
approx_contours = [cv2.approxPolyDP(cnt, 2, True) for cnt in contours if cv2.contourArea(cnt) >= 0.1]
mask = cv2.drawContours(np.zeros(map.shape, np.uint8), approx_contours, -1, 1, -1)
actual = cv2.imread(os.path.join(folder_mask, maps[0]), cv2.IMREAD_GRAYSCALE)
actual[actual <= 40] = 0
actual[actual > 0] = 1
cv2.imwrite(os.path.join(folder_poly, 'map.png'), map * 255)
cv2.imwrite(os.path.join(folder_poly, 'actual.png'), actual * 255)
tp += ((mask == 1) & (actual == 1)).sum()
fp += ((mask == 1) & (actual == 0)).sum()
fn += ((mask == 0) & (actual == 1)).sum()
area_pred += mask.sum()
area_fact += actual.sum()

pre = tp / (tp + fp)
rec = tp / (tp + fn)
f1 = 2 * pre * rec / (pre + rec)

print('f1 = {}'.format(round(f1, 3)))
print('jacard = {}'.format(round(tp / (tp + fp + fn), 3)))
print('area error = {}'.format(round((area_pred - area_fact) / area_fact, 3)))




