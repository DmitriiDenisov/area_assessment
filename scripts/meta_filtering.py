import cv2
import os
import numpy as np
from area_assesment.geo.utils import write_geotiff


model_names = ['color_rf10', 'cshapes']
# output_folder = os.path.normpath('../data/output/')
sat_folder = os.path.normpath('/storage/_pdata/sakaka/satellite_images/raw_geotiffs/Area_Sakaka_Dawmat_Al_Jandal/')
output_folder = os.path.normpath('../data/output/test/')
nchannels_sat = 3

heat_files = sorted([f for f in os.listdir(output_folder)
                     if os.path.isfile(os.path.join(output_folder, f)) and f[:5] == 'mean_'])
sat_files = sorted([f for f in os.listdir(sat_folder) if os.path.isfile(os.path.join(sat_folder, f))])
nheats = len(heat_files)
nmodels = len(model_names)

thresh1 = 75
thresh2 = 175
ksize = 51


for i in range(nheats):
    print('Image #{}/{}'.format(i + 1, nheats))
    heat = cv2.imread(os.path.join(output_folder, heat_files[i]), cv2.IMREAD_GRAYSCALE)
    heat[heat < thresh1] = 0
    heat = cv2.medianBlur(heat, ksize)
    heat[heat < thresh2] = 0
    write_geotiff(os.path.join(output_folder, 'heatmap_' + heat_files[i][5:-4]), raster_layers=heat,
                  gdal_ds=os.path.join(sat_folder, sat_files[i]))
