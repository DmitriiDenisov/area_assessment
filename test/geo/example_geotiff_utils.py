import os
import gdal
import cv2

from area_assesment.geo.utils import write_geotiff

## Create and write GeoTIFF

ref_geotiff = os.path.normpath("../../data/test/geo/Dawmat_Al_Jandal_01_09.tif")
mask = os.path.normpath("../../data/test/geo/Dawmat_Al_Jandal_01_09_Mask.tif")
dst_path = os.path.normpath("../../data/test/res_artifacts/Dawmat_Al_Jandal_01_09_Mask_Geotiff.tif")

# read image as np.array
img = cv2.imread(mask, cv2.IMREAD_COLOR)

# read geotiff as GDAL Dataset to copy metadata from
src_ds = gdal.Open(ref_geotiff)

# create GeoTIFF from image with all necessary meta
write_geotiff(dst_path, img, src_ds)
