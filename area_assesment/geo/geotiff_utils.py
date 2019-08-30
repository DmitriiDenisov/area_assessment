import numpy as np
import gdal
import logging


def write_geotiff(path: str, raster_layers: np.array, gdal_ds: str):
    logging.debug('WRITING GEOTIFF')
    gdal_ds = gdal.Open(gdal_ds)
    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(path,
                               raster_layers.shape[1],
                               raster_layers.shape[0],
                               len(raster_layers.shape),
                               gdal.GDT_Byte)

    out_raster.SetMetadata(gdal_ds.GetMetadata())
    out_raster.SetGeoTransform(gdal_ds.GetGeoTransform())
    out_raster.SetProjection(gdal_ds.GetProjection())

    if len(raster_layers.shape) == 3:
        for channel_num in range(raster_layers.shape[2]):
            # reverse array so the tif looks like the array
            raster_layer = raster_layers[:, :, channel_num]
            outband = out_raster.GetRasterBand(channel_num + 1)
            outband.WriteArray(raster_layer)
    else:
        out_raster.GetRasterBand(1).WriteArray(raster_layers)

    out_raster.FlushCache()
    logging.info('GEOTIFF SAVED: {}'.format(path))


def write_geotiff_cut_image(output_file, gdal_ds, new_image, row_number, col_number, path_size):
    # '../../data/train/big_sat/mecca_google.ешаэ
    # gdal_ds = gdal.Open(path_file)

    transform = gdal_ds.GetGeoTransform()
    xOrigin = transform[0]
    yOrigin = transform[3]
    pixelWidth = transform[1]
    pixelHeight = transform[5]
    new_x = xOrigin + col_number * path_size[0] * pixelWidth  # !!!!!!!!
    new_y = yOrigin + row_number * path_size[1] * pixelHeight  # !!!!!!!!!


    driver = gdal.GetDriverByName('GTiff')
    out_raster = driver.Create(output_file,
                               new_image.shape[1],
                               new_image.shape[0],
                               len(new_image.shape),
                               gdal.GDT_Byte)

    out_raster.SetMetadata(gdal_ds.GetMetadata())
    new_transform = (new_x, pixelWidth, transform[2], new_y, transform[4], pixelHeight)
    out_raster.SetGeoTransform(new_transform)
    out_raster.SetProjection(gdal_ds.GetProjection())

    if len(new_image.shape) == 3:
        for channel_num in range(new_image.shape[2]):
            # reverse array so the tif looks like the array
            raster_layer = new_image[:, :, channel_num]
            outband = out_raster.GetRasterBand(channel_num + 1)
            outband.WriteArray(raster_layer)
    else:
        out_raster.GetRasterBand(1).WriteArray(new_image)

    out_raster.FlushCache()

