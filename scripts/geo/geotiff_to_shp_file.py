import argparse
import os
import glob
import sys
import cv2
import gdal
from math import floor
from gdal import ogr, osr
from gdalconst import GA_ReadOnly
from shapely.geometry import Polygon, MultiPolygon

from area_assesment.images_processing.polygons import mask_to_polygons

parser = argparse.ArgumentParser(description='Creates ESRI shapefile from GeoTIFFs images with mask.')
parser.add_argument('src_dir', metavar='src_dir', type=str,
                    help='source file or files mask')
parser.add_argument('dst_dir', metavar='dst_dir', type=str,
                    help='destination shape file path')
parser.add_argument('-v', '--verbose', dest='v', action='store_const',
                    const=True, default=False,
                    help='print the processing details')
parser.add_argument('-l', '--layer_name', dest='ln', type=str,
                    default='default_layer',
                    help='layer name')
parser.add_argument('-f', '--feature_name', dest='fn', type=str,
                    default='feature',
                    help='feature name')
parser.add_argument('--target_EPSG', dest='t_epsg', type=int,
                    help='EPSG code to reproject geometry')
parser.add_argument('--mask_threshold', dest='m_thre', type=float,
                    default=0.5,
                    help='threshold to for separation of polygons from background')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

gdal.AllRegister()

src_dir = os.path.abspath(os.path.normpath(args.src_dir))
dst_dir = os.path.abspath(os.path.normpath(args.dst_dir))

polygons_list = []

projection_wkt = None

for f in [os.path.basename(f) for f in glob.glob(src_dir) if os.path.isfile(os.path.join(src_dir, f))]:
    cur_file_path = os.path.join(os.path.dirname(src_dir), f)
    mask = cv2.imread(cur_file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask[mask < floor(args.m_thre * 255)] = 0
    mask[mask >= floor(args.m_thre * 255)] = 1

    image_polygons = mask_to_polygons(mask, epsilon=1, min_area=0.02, rect_polygon=False)

    gdal_ds = gdal.Open(cur_file_path, GA_ReadOnly)
    top_left_x = gdal_ds.GetGeoTransform()[0]
    top_left_y = gdal_ds.GetGeoTransform()[3]
    x_resolution = gdal_ds.GetGeoTransform()[1]
    y_resolution = gdal_ds.GetGeoTransform()[5]

    for p in image_polygons:
        x = [(top_left_x + x_resolution * x) for x in p.exterior.coords.xy[0]]
        # yes gdal_ds.RasterXSize for vertical axis, don't know why yet
        y = [(top_left_y - y_resolution * (y - gdal_ds.RasterXSize)) for y in p.exterior.coords.xy[1]]
        p_transformed = Polygon(list(zip(x, y)))
        polygons_list.append(p_transformed)

    # gets projection wkt of the last processed file
    projection_wkt = gdal_ds.GetProjection()

mult_p = MultiPolygon(polygons_list)

driver = ogr.GetDriverByName('ESRI Shapefile')
driver_c = ogr.GetDriverByName('ESRI Shapefile')

ds = driver.CreateDataSource(os.path.join(dst_dir,"{}.shp".format(args.ln)))
ds_c = driver_c.CreateDataSource(os.path.join(dst_dir,"{}-centroids.shp".format(args.ln)))

source_srs = osr.SpatialReference()
source_srs.ImportFromWkt(projection_wkt)
srs = source_srs

if (args.t_epsg is not None):
    target_srs = osr.SpatialReference()
    target_srs.ImportFromEPSG(args.t_epsg)
    transform = osr.CoordinateTransformation(source_srs, target_srs)
    srs = target_srs
else:
    transform = osr.CoordinateTransformation(source_srs, source_srs)

# create new layer definition and define columns
# of attribute table
layer = ds.CreateLayer(args.ln, srs, ogr.wkbMultiPolygon)
layer.CreateField(ogr.FieldDefn("name", ogr.OFTString))
layer.CreateField(ogr.FieldDefn("key", ogr.OFTString))
layer.CreateField(ogr.FieldDefn("area", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("totalArea", ogr.OFTReal))
layer.CreateField(ogr.FieldDefn("totalCount", ogr.OFTInteger))

layer_c = ds_c.CreateLayer(args.ln+"_centroids", srs, ogr.wkbPoint)
layer_c.CreateField(ogr.FieldDefn("name", ogr.OFTString))
layer_c.CreateField(ogr.FieldDefn("area", ogr.OFTReal))

total_area = 0.0

for i, p in enumerate(mult_p):
    # create bew feature
    feature = ogr.Feature(layer.GetLayerDefn())
    # create geometry from polygon
    geom_poly = ogr.CreateGeometryFromWkb(p.wkb)
    # transform geometry from source_srs to target_srs
    geom_poly.Transform(transform)
    # set the feature geometry and attributes
    feature.SetGeometry(geom_poly)
    feature.SetField("area", geom_poly.GetArea())
    total_area = total_area + geom_poly.GetArea()
    feature.SetField("name", "{}_{}".format(args.fn, i))
    feature.SetField("key", "{}_{}".format(args.fn, i))
    feature.SetField("totalCount", len(mult_p))
    # add the feature in the layer
    layer.CreateFeature(feature)
    # Dereference the feature

    #
    feature_c = ogr.Feature(layer_c.GetLayerDefn())
    centroid = geom_poly.Centroid()
    feature_c.SetGeometry(centroid)
    feature_c.SetField("area", geom_poly.GetArea())
    feature_c.SetField("name", "{}_{}".format(args.fn, i))

    layer_c.CreateFeature(feature_c)

    feature_c = None
    feature = None

# terrible workaround to store total area and count of features of mapbox
for i in range(layer.GetFeatureCount()):
    feature = layer.GetFeature(i)
    feature.SetField("totalArea", total_area)
    layer.SetFeature(feature)


# Save and close everything
ds = layer = feat = geom = None
ds_c = layer_c = None