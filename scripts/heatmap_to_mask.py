import numpy as np
import cv2
import sys
from area_assesment.images_processing.contours import convex_contours
from area_assesment.images_processing.polygons import contours_to_polygons, plot_polygons, save_polygons

sys.path.append('../area_assesment/images_processing')


folder_in = '../sakaka_data/heatmap/'
folder_out = '../sakaka_data/map2/'

contour_list, files, shapes = convex_contours(folder_in, folder_out, min_area=10, cutoff=36)
polygon_list = contours_to_polygons(contour_list, shapes, tolerance=2)

for i in range(len(files)):
    print('Plotting image #{i} out of {n}'.format(i=i + 1, n=len(files)))
    plot_polygons(polygon_list[i], 'polygons_' + files[i], output_folder=folder_out)
    save_polygons(polygon_list[i], output_folder=folder_out, fname=files[i])





