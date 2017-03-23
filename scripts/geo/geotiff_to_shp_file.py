import argparse
import os
import glob
import sys
import cv2
import gdal
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
                    default='default',
                    help='layer name')

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
    mask[mask > 0] = 1

    image_polygons = mask_to_polygons(mask)

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

srs = osr.SpatialReference()
srs.ImportFromWkt(projection_wkt)

layer = ds.CreateLayer(args.ln, srs, ogr.wkbMultiPolygon)

# Add one attribute
layer.CreateField(ogr.FieldDefn(args.ln, ogr.OFTInteger))
defn = layer.GetLayerDefn()

# Create a new feature (attribute and geometry)
feat = ogr.Feature(defn)
feat.SetField(args.ln, 1)

# Make a geometry, from Shapely object
geom = ogr.CreateGeometryFromWkb(mult_p.wkb)
feat.SetGeometry(geom)

layer.CreateFeature(feat)

feat = geom = None  # destroy these

# Save and close everything
ds = layer = feat = geom = None

print()
