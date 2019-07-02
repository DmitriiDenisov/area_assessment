import numpy as np
import cv2
from os import listdir
from os.path import isfile, join
import sys
sys.path.append('../area_assesment/images_processing/')
from polygons import mask_to_polygons, plot_polygons, save_polygons
import gdal
from gdal import ogr, osr


path = '../data/maps'
files = [f for f in listdir(path) if isfile(join(path, f))]
nfiles = len(files)
poly = list(range(nfiles))
meta = list(range(nfiles))
gdal.AllRegister()

for f in range(nfiles):
    print('File #{f} out of {n}'.format(f=f + 1, n=nfiles))
    mask = cv2.imread('{p}/{f}'.format(p=path, f=files[f]), cv2.IMREAD_GRAYSCALE)
    mask[mask > 0] = 1
    # mask = cv2.flip(mask, 0)
    cv2.imwrite('../../data/output/' + files[f], mask * 255)
    poly[f] = mask_to_polygons(mask)
    plot_polygons(poly[f], 'polygons_{}'.format(files[f][:-4]))
    # md = gdal.Open('{p}/{f}'.format(p=path, f=files[f])).GetMetadata()
    # meta[f] = dict(wkt=md['IMAGE__WKT'], xy_origin=md['IMAGE__XY_ORIGIN'], x_resolution=md['IMAGE__X_RESOLUTION'],
    #                y_resolution=md['IMAGE__Y_RESOLUTION'])
    save_polygons(poly[f], 'polygons_{}'.format(files[f][:-4]))





