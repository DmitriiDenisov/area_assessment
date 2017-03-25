import numpy as np
import gdal
import logging


def write_geotiff(path: str, raster_layers: np.array, gdal_ds: gdal.Dataset):
    logging.info('WRITING GEOTIFF')
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
