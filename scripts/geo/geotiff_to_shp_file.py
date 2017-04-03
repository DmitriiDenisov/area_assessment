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
parser.add_argument('dst_shp', metavar='dst_shp', type=str,
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

src_dir = os.path.normpath(args.src_dir)
dst_dir = os.path.normpath(args.dst_shp)

polygons_list = []

projection_wkt = None

for f in [os.path.basename(f) for f in glob.glob(src_dir) if os.path.isfile(os.path.join(src_dir, f))]:
    cur_file_path = os.path.join(os.path.dirname(src_dir), f)
    mask = cv2.imread(cur_file_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        continue
    mask[mask < floor(0.1 * 255)] = 0
    mask[mask >= floor(0.1 * 255)] = 1

    image_polygons = mask_to_polygons(mask, epsilon=2, min_area=0.2, rect_polygon=True)

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
ds = driver.CreateDataSource(args.dst_shp)

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
layer.CreateField(ogr.FieldDefn("Name", ogr.OFTString))
layer.CreateField(ogr.FieldDefn("Area", ogr.OFTReal))

for i, p in enumerate(mult_p):
    # create bew feature
    feature = ogr.Feature(layer.GetLayerDefn())
    # create geometry from polygon
    geom_poly = ogr.CreateGeometryFromWkb(p.wkb)
    # transform geometry from source_srs to target_srs
    geom_poly.Transform(transform)
    # set the feature geometry and attributes
    feature.SetGeometry(geom_poly)
    feature.SetField("Area", geom_poly.GetArea())
    feature.SetField("Name", "{}_{}".format(args.fn, i))
    # add the feature in the layer
    layer.CreateFeature(feature)
    # Dereference the feature
    feature = None

# Save and close everything
ds = layer = feat = geom = None
