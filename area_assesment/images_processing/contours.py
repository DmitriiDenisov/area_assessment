import numpy as np
import cv2
from os import listdir
from os.path import isfile, join


def convex_contours(folder_in='../data/heatmaps', folder_out='../data/maps', min_area=10, cutoff=30):

    files = [f for f in listdir(folder_in) if isfile(join(folder_in, f))]
    nfiles = len(files)
    contour_list = []
    shapes = []

    for i in range(nfiles):
        print('Contouring image #{i} out of {n}'.format(i=i + 1, n=nfiles))
        img = cv2.imread(join(folder_in, files[i]), cv2.IMREAD_GRAYSCALE)
        shapes.append(img.shape)
        ret, thresh = cv2.threshold(img, cutoff, 255, 0)
        # finding contours
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # for k in range(len(contours)):
        #     contours[k] = cv2.convexHull(contours[k])
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        cv2.imwrite(join(folder_out, files[i]), thresh)
        # drawing filled contours
        img = np.zeros(img.shape, np.uint8)
        cv2.drawContours(img, contours, -1, 255, -1)
        # repeat finding contours to merge child contours (within parent contours) with parent contours
        ret, thresh = cv2.threshold(img, cutoff, 255, 0)
        im2, contours, hierarchy = cv2.findContours(thresh, 1, 2)
        img = np.zeros(img.shape, np.uint8)
        cv2.drawContours(img, contours, -1, 255, -1)
        cv2.imwrite(join(folder_out, 'cont_' + files[i]), img)
        contour_list.append(contours)

    return contour_list, files, shapes
