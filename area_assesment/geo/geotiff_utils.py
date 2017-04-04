import numpy as np
import gdal
import logging

from shapely.geometry import Polygon


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


def image_coords_to_geo(image_polygons, geo_transform, raster_x_size ):
    ret = []

    top_left_x = geo_transform[0]
    top_left_y = geo_transform[3]
    x_resolution = geo_transform[1]
    y_resolution = geo_transform[5]

    for p in image_polygons:
        x = [(top_left_x + x_resolution * x) for x in p.exterior.coords.xy[0]]
        # yes gdal_ds.RasterXSize for vertical axis, don't know why yet
        y = [(top_left_y - y_resolution * (y - raster_x_size)) for y in p.exterior.coords.xy[1]]
        ret.append(Polygon(list(zip(x, y))))

    return ret
