import cv2
import os
import numpy as np


err_thresh = np.inf
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
    cont_new[cont_new < 0] = 0
    return cont_new.astype(np.int32)


crops_sat = []
crops_mask = []
count = 0
print('Collecting marked buildings...')
for i in range(nsats):
    sat = cv2.imread(os.path.join(sat_folder, sat_files[i]), cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(os.path.join(map_folder, map_files[i]), cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 255
    cont = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(cont) > 0:
        for k in range(len(cont)):
            count += 1
            crop = sat[cont[k][:, 0, 1].min():(cont[k][:, 0, 1].max() + 1),
                       cont[k][:, 0, 0].min():(cont[k][:, 0, 0].max() + 1)].copy()
            crop_mask = mask[cont[k][:, 0, 1].min():(cont[k][:, 0, 1].max() + 1),
                             cont[k][:, 0, 0].min():(cont[k][:, 0, 0].max() + 1)].copy()
            crops_sat.append(crop)
            crops_mask.append(crop_mask)

print('Processing images...')
ncrops = len(crops_sat)
for i in range(nheats):
    sat = cv2.imread(os.path.join(sat2_folder, heat_files[i][8:]), cv2.IMREAD_GRAYSCALE)
    heat = cv2.imread(os.path.join(heat_folder, heat_files[i]), cv2.IMREAD_GRAYSCALE)
    heat[heat < 125] = 0
    heat[heat > 0] = 255
    cont = cv2.findContours(heat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
    if len(cont) > 0:
        for k in range(len(cont)):
            print('Image {}/{}, contour #{}/{}'.format(i + 1, nheats, k + 1, len(cont)))
            cont[k] = resize_cont(cont[k], coef)
            crop = sat[cont[k][:, 0, 1].min():(cont[k][:, 0, 1].max() + 1),
                       cont[k][:, 0, 0].min():(cont[k][:, 0, 0].max() + 1)].copy()
            # hist = cv2.calcHist([crop], [0], None, [64], [0, 256]).flatten()
            err = np.array([((crop - cv2.resize(crops_sat[j], (crop.shape[1], crop.shape[0]))) ** 2).sum()
                            for j in range(ncrops)], np.float64)
            if err.min() < err_thresh:
                heat[cont[k][:, 0, 1].min():(cont[k][:, 0, 1].max() + 1),
                     cont[k][:, 0, 0].min():(cont[k][:, 0, 0].max() + 1)] = \
                    cv2.resize(crops_mask[err.argmin()], (crop.shape[1], crop.shape[0]))
    cv2.imwrite(os.path.join(output_folder, 'corr_' + heat_files[i][8:]), heat)




