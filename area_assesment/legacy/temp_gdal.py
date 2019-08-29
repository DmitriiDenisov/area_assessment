import numpy as np
import gdal
import logging
from osgeo import osr
import numpy as np
from gdalconst import GA_ReadOnly
from keras_preprocessing.image import load_img
import gdal

img = np.array(load_img('../../data/train/big_sat/mecca_google.tif', grayscale=False))
gdal_ds = gdal.Open('../../data/train/big_sat/mecca_google.tif')
# gdal_ds = gdal.Open('new.tif')

# gdal.Translate('../../data/train/big_sat/Mecca.tif', 'input.tif', projWin=[-100, 100, 100, -200])

# gdal.Translate('new.tif', gdal_ds, projWin=[0, 0, 100, -150])

a = gdal_ds.GetMetadata()
b = gdal_ds.GetGeoTransform()
c = gdal_ds.GetProjection()
# print(a)
# print(b)
# print(c)
transform = gdal_ds.GetGeoTransform()
xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = transform[5]
new_x = xOrigin + 2 * 512 * pixelWidth #!!!!!!!!
new_y = yOrigin + 6 * 512 * pixelHeight #!!!!!!!!!
new_img = img[6*512:7*512, 2*512:4*512].copy()



driver = gdal.GetDriverByName('GTiff')
out_raster = driver.Create('new.tif',
                           new_img.shape[1],
                           new_img.shape[0],
                           len(new_img.shape),
                           gdal.GDT_Byte)

out_raster.SetMetadata(gdal_ds.GetMetadata())
new_transform = (new_x, pixelWidth, transform[2], new_y, transform[4], pixelHeight)
out_raster.SetGeoTransform(new_transform)
out_raster.SetProjection(gdal_ds.GetProjection())

if len(new_img.shape) == 3:
    for channel_num in range(new_img.shape[2]):
        # reverse array so the tif looks like the array
        raster_layer = new_img[:, :, channel_num]
        outband = out_raster.GetRasterBand(channel_num + 1)
        outband.WriteArray(raster_layer)
else:
    out_raster.GetRasterBand(1).WriteArray(new_img)

out_raster.FlushCache()




1 / 0


transform = gdal_ds.GetGeoTransform()
band = gdal_ds.GetRasterBand(1)

xOrigin = transform[0]
yOrigin = transform[3]
pixelWidth = transform[1]
pixelHeight = - transform[5]
p1 = (4429968, 2443270)
p2 = (4429995, 2443250)

i1 = int((p1[0] - xOrigin) / pixelWidth)
j1 = int((yOrigin - p1[1] ) / pixelHeight)
i2 = int((p2[0] - xOrigin) / pixelWidth)
j2 = int((yOrigin - p2[1]) / pixelHeight)


new_cols = i2-i1+1
new_rows = j2-j1+1

data = band.ReadAsArray(i1, -j1, new_cols, new_rows)

new_x = xOrigin + i1*pixelWidth #!!!!!!!!
new_y = yOrigin - j1*pixelHeight #!!!!!!!!!

new_transform = (new_x, transform[1], transform[2], new_y, transform[4], transform[5])

driver = gdal.GetDriverByName("GTiff")
output_file = "raster2.tif"
dst_ds = driver.Create(output_file,
                       new_cols,
                       new_rows,
                       1,
                       gdal.GDT_Float32)
dst_ds.GetRasterBand(1).WriteArray( data )
dst_ds.SetGeoTransform(new_transform)
wkt = gdal_ds.GetProjection()

# setting spatial reference of output raster
srs = osr.SpatialReference()
srs.ImportFromWkt(wkt)
dst_ds.SetProjection(srs.ExportToWkt())
#Close output raster dataset
dataset = None
dst_ds = None
