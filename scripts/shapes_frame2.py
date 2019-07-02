import cv2
import os
import numpy as np


coef = 0.25

sat_folder = os.path.normpath('/storage/_pdata/sakaka/satellite_images/retiled_geotiffs/Dawmat_Al_Jandal/0/')
map_folder = os.path.normpath('/storage/_pdata/sakaka/satellite_images/retiled_geotiffs/Dawmat_Al_Jandal/0_map/')
sat2_folder = os.path.normpath('../data/contours/sat/')
heat_folder = os.path.normpath('../data/contours/heat/')
output_folder = os.path.normpath('../data/contours/cont/')

sat_files = sorted([f for f in os.listdir(sat_folder) if os.path.isfile(os.path.join(sat_folder, f))])
map_files = sorted([f for f in os.listdir(map_folder) if os.path.isfile(os.path.join(map_folder, f))])
heat_files = sorted([f for f in os.listdir(heat_folder) if os.path.isfile(os.path.join(heat_folder, f))])
nsats = len(sat_files)
nheats = len(heat_files)

def resize_cont(cont, coef):
    mu = cont[:, 0, :].mean(0)
    cont_new = cont.copy() * (1 + coef)
    cont_new[:, 0, :] += (mu - cont_new[:, 0, :].mean(0)).reshape(1, -1)
    return cont_new.astype(np.int32)


print('Processing images...')
for i in range(nheats):
    sat = cv2.imread(os.path.join(sat2_folder, heat_files[i][8:])).astype(np.uint8)
    heat = cv2.imread(os.path.join(heat_folder, heat_files[i]), cv2.IMREAD_GRAYSCALE)#.astype(np.int32)
    heat[heat < 125] = 0
    heat[heat > 0] = 255
    cont, h = cv2.findContours(heat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:3]
    corr = np.zeros(heat.shape, np.int32)
    heat_color = np.zeros((heat.shape[0], heat.shape[1], 3), np.uint8)
    for k in range(3):
        heat_color[:, :, k] = heat.copy()
    if len(cont) > 0:
        for k in range(len(cont)):
            print('Image {}/{}, contour #{}/{}'.format(i + 1, nheats, k + 1, len(cont)))
            if (cv2.contourArea(cont[k]) >= 50) and (h[0, k, 3] == -1):
                rect = cv2.minAreaRect(resize_cont(cont[k], coef))
                box = cv2.boxPoints(rect)
                print(box.shape)
                cnt = np.array((box.shape[0], 1, box.shape[1]), np.uint8)
                print(cnt.shape)
                cnt[:, 0, :] = box.copy()
                # height = box[:, 0].max() - box[:, 0].min()
                # width = box[:, 1].max() - box[:, 1].min()
                # box[0] = [int(box[0][0] + 0.2 * height), int(box[0][1] + 0.2 * width)]
                # box[1] = [int(box[1][0] - 0.2 * height), int(box[1][1] + 0.2 * width)]
                # box[2] = [int(box[2][0] - 0.2 * height), int(box[2][1] - 0.2 * width)]
                # box[3] = [int(box[3][0] + 0.2 * height), int(box[3][1] - 0.2 * width)]
                cv2.drawContours(sat, cnt, -1, (255, 0, 0), -1)
    cv2.imwrite(os.path.join(output_folder, 'corr_' + heat_files[i][8:]), sat)

