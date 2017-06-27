import numpy as np
import cv2
import sys
import os
from os import listdir
from os.path import isfile, join
from area_assesment.images_processing.contours import convex_contours
from area_assesment.images_processing.polygons import contours_to_polygons, plot_polygons, save_polygons, \
    mask_to_polygons
from area_assesment.io_operations.data_io import filenames_in_dir

sys.path.append('../area_assesment/images_processing')


folder_in = os.path.normpath('../sakaka_data/output/sakaka_test/')
folder_out = os.path.normpath('../sakaka_data/output/sakaka_test/sakaka_test_polygons/')

contour_list, files, shapes = convex_contours(folder_in, folder_out, min_area=10, cutoff=36)
# polygon_list = contours_to_polygons(contour_list, shapes, tolerance=2)

# files = [f for f in listdir(folder_in) if isfile(join(folder_in, f))]
files = filenames_in_dir(folder_in, endswith_='.tif')
nfiles = len(files)
contour_list = []
shapes = []
polygon_list = []
for i in range(nfiles):
    print('Contouring image #{i} out of {n}'.format(i=i + 1, n=nfiles))
    img = cv2.imread(join(folder_in, files[i]), cv2.IMREAD_GRAYSCALE)
    polygon_list.append(mask_to_polygons(img, epsilon=1, min_area=0.02))

for i in range(len(files)):
    print('Plotting image #{i} out of {n}'.format(i=i + 1, n=len(files)))
    plot_polygons(polygon_list[i], 'polygons_' + files[i], output_folder=folder_out)
    save_polygons(polygon_list[i], output_folder=folder_out, fname=files[i])





