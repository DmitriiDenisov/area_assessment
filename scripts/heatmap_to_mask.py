import numpy as np
import cv2
import sys
sys.path.append('../area_assesment/images_processing')
from contours import convex_contours
from polygons import contours_to_polygons, save_polygons, plot_polygons


folder_in = '../data/heatmaps'
folder_out = '../data/maps'

contour_list, files, shapes = convex_contours(folder_in, folder_out, min_area=10, cutoff=36)
polygon_list = contours_to_polygons(contour_list, shapes, tolerance=2)

for i in range(len(files)):
    print('Plotting image #{i} out of {n}'.format(i=i + 1, n=len(files)))
    plot_polygons(polygon_list[i], 'polygons_' + files[i])
    save_polygons(polygon_list[i], files[i])





