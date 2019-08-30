import os

import cv2
from pyproj import Proj, transform
import json
import gdal
from gdalconst import GA_ReadOnly
from area_assesment.io_operations.visualization import plot2
from PIL import Image, ImageDraw
from skimage.draw import polygon
import numpy as np

# Скрипт читает изображения из папки source_dir. Для каждого изображения создается изображение в папке
# target_dir с такими же размерами и таким же названием + '_MAP'

def convert_coords_norm(dict_new_polygons, geo_transform):
    # Функция, которая переводит координаты из geo в относительные
    top_left_x = geo_transform[0]
    top_left_y = geo_transform[3]
    x_resolution = geo_transform[1]
    y_resolution = geo_transform[5]

    dict_norm_polygons = {}
    for id, poly in dict_new_polygons.items():
        dict_norm_polygons[id] = []
        for (x_epsg, y_epsg) in poly:
            x_norm = (x_epsg - top_left_x) / x_resolution
            # y_norm = raster_x_size + (top_left_y - y_epsg) / y_resolution
            y_norm = -(top_left_y - y_epsg) / y_resolution
            dict_norm_polygons[id].append([x_norm, y_norm])
    return dict_norm_polygons


source_dir = '../../data/train/big_sat'
target_dir = '../../data/train/map'
P3857 = Proj(init='epsg:3857')
P4326 = Proj(init='epsg:4326')
# Прочитать json
guestFile = open("../../data/train/features.geojson", 'r')
guestData = guestFile.read()
guestFile.close()
gdfJson = json.loads(guestData)

list_of_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) and f[-4:] == '.tif']
print('Total number of files: {}'.format(len(list_of_files)))
for f in list_of_files:
    print('Processing for file:{}'.format(f))
    cur_file_path = os.path.join(source_dir, f)
    # Прочитать изображение
    img_sat = cv2.imread(cur_file_path)

    gdal_ds = gdal.Open(cur_file_path, GA_ReadOnly)
    geo_transform = gdal_ds.GetGeoTransform()
    raster_x_size = gdal_ds.RasterXSize
    projection_wkt = gdal_ds.GetProjection()

    # Перевести координаты long/lat в EPSG:3857
    dict_new_polygons = {}
    for poly_dict in gdfJson['features']:
        id = poly_dict['id']
        converted_coords = []
        for (lon, lat) in poly_dict['geometry']['coordinates'][0]:
            x, y = transform(P4326, P3857, lon, lat)
            converted_coords.append([x, y])
        dict_new_polygons[id] = converted_coords[:]

    # Получить из относительных координат маску
    dict_norm_polygons = convert_coords_norm(dict_new_polygons, gdal_ds.GetGeoTransform())

    # Сохранить маску
    width = img_sat.shape[1]
    height = img_sat.shape[0]

    new_mask = np.zeros((height, width)).astype(np.uint8)
    for key, polygon_ in dict_norm_polygons.items():
        polygon_ = [(x[0], x[1]) for x in polygon_]
        r = np.array([x[1] for x in polygon_])
        c = np.array([x[0] for x in polygon_])

        rr, cc = polygon(r, c)
        try:
            new_mask[rr, cc] = 1
        except:
            pass

    im = Image.fromarray(new_mask * 255)
    path_ = os.path.join(target_dir, "{}_MAP.tif".format(f[:-4]))
    im.save(path_)
    # plot2(img_sat, new_mask, name='{}_NEW_MASK'.format(f),
    #      overlay=True, alpha=0.5, show_plot=False, save_output_path='')

# Just another option how to save mask:
# x, y = np.meshgrid(np.arange(width), np.arange(height))
# x, y = x.flatten(), y.flatten()
#
# points = np.vstack((x, y)).T
#
# grid_all = np.zeros((height, width))
# for key, polygon in dict_norm_polygons.items():
#     polygon = [(x[0], x[1]) for x in polygon]
#     path = Path(polygon)
#     grid = path.contains_points(points)
#     grid = grid.reshape((height, width))
#     grid_all += grid
#
# im = Image.fromarray((grid_all * 255).astype(np.uint8))
# im.save("your_file_4.jpeg")
#
# plot2(img_sat, grid_all, name='GRID_ALL',
#       overlay=True, alpha=0.5, show_plot=False, save_output_path='')