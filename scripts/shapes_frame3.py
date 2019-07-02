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


print('Downloading sample maps...')
sats = list(range(nsats))
maps = list(range(nsats))
for i in range(nsats):
    sats[i] = cv2.imread(os.path.join(sat_folder, sat_files[i]))
    maps[i] = cv2.imread(os.path.join(map_folder, map_files[i]), cv2.IMREAD_GRAYSCALE)

print('Processing images...')
for i in range(nheats):
    print('    Image {}/{}'.format(i + 1, nheats))
    nconts = 0
    x = []; y = []; buildings = []
    sat = cv2.imread(os.path.join(sat2_folder, heat_files[i][8:]))
    heat = cv2.imread(os.path.join(heat_folder, heat_files[i]), cv2.IMREAD_GRAYSCALE)
    heat[heat < 125] = 0
    heat[heat > 0] = 255
    print('        Finding contours...')
    cont, h = cv2.findContours(heat, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1:3]
    if len(cont) > 0:
        for k in range(len(cont)):
            if (cv2.contourArea(cont[k]) >= 50) and (h[0, k, 3] == -1):
                box = cv2.boundingRect(cont[k])
                x.append(box[0])
                y.append(box[1])
                buildings.append(sat[box[0]:(box[0] + box[2]), box[1]:(box[1] + box[3]), :].copy())
                nconts += 1
    print('        Generating probability maps...')
    probs = list(range(nconts))
    mi = np.inf; ma = -np.inf
    for j in range(nconts):
        print('            Contour #{}/{}'.format(j + 1, nconts))
        probs.append(cv2.matchTemplate(sat, buildings[j], cv2.TM_SQDIFF_NORMED))
        mi = min((mi, probs[-1].min()))
        ma = max((ma, probs[-1].max()))
    print(mi, ma)
    raise SystemExit
