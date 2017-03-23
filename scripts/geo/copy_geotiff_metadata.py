import argparse
import glob
import os
import gdal
from gdalconst import GA_Update
import sys

parser = argparse.ArgumentParser(description='Copies metadata from GeoTIFFs in one folder to GeoTIFFs in another'
                                             'with the same filename.')
parser.add_argument('src_dir', metavar='src_dir', type=str,
                    help='source directory')
parser.add_argument('dest_dir', metavar='dest_dir', type=str,
                    help='target directory')
parser.add_argument('-v', '--verbose', dest='v', action='store_const',
                    const=True, default=False,
                    help='print the processing details')

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)

src_dir = os.path.normpath(args.src_dir)
dst_dir = os.path.normpath(args.dest_dir)

for f in [os.path.basename(f) for f in glob.glob(src_dir) if os.path.isfile(os.path.join(src_dir, f))]:
    src_path = os.path.join(os.path.dirname(src_dir), f)
    dst_path = os.path.join(dst_dir, f)

    if args.v:
        print('Processed: {}'.format(f))

    if not os.path.isfile(os.path.join(dst_dir, f)):
        print('No file to write metadata in the destination dir: {}'.format(f))
        continue

    src_ds = gdal.Open(src_path)
    dst_ds = gdal.Open(dst_path, GA_Update)

    dst_ds.SetMetadata(src_ds.GetMetadata())
    dst_ds.SetGeoTransform(src_ds.GetGeoTransform())
    dst_ds.SetProjection(src_ds.GetProjection())
    # writes file to disk
    dst_ds.FlushCache()
    dst_ds = None
    src_ds = None
    dest_ds = None
