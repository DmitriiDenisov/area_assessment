# area_assessment

###

<p align="center">
  <img src="https://media.giphy.com/media/cPHRdYuWdXOW6AtYjd/giphy.gif" width="600" alt="accessibility text">
</p>

### Screenshots:
<p align="center">
  <img src="https://i.ibb.co/fv5fTkM/Screen-Shot-2019-10-10-at-01-33-00.png" width="900" alt="accessibility text">
</p>

<p align="center">
  <img src="https://i.ibb.co/hX8kChQ/Screen-Shot-2019-10-10-at-01-31-35.png" width="900" alt="accessibility text">
</p>

<p align="center">
  <img src="https://i.ibb.co/PcnvvJD/Screen-Shot-2019-10-10-at-01-31-22.png" width="900" alt="accessibility text">
</p>

### Instructions:
0. **Create dataset**. With SAS.Planet (SAS.Планета) parse high resolution images. Recommendation is to use zoom 18, because >18 is too delatiled and <18 is not enough. With ArcGIS Online create geojson with coordinates of buildings
1. **Create masks for sat images** `area_assessment/make_train_masks.py`: script reads files from `source_dir` (default: `../../data/train/sat`). For each image from `source_dir` this script makes corresponding binary mask image in `target_dir` (default `../../data/train/map`). In order to make it script takes as input `geojson_path` variable (default: `../../data/train/NN_predict_0_geojson_1.geojson`)
First of all it reads `geojson_path` file and converts coordinates from P4326 to P3857 system and saves new coordinates into new json file because it takes a lot time for converting. Then it takes every image from `source_dir`, in loop over every polygon from `geojson_path` file creates binary mask for an image and saves it in `target_dir`
2. **Train model**. `scripts/train.py`: calls `DataGeneratorCustom` from `area_assessment/neural_networks/DataGeneratorCustom` which reads every image from folder with sat images (default `../data/train/sat`) and corresponding masks which we obtained on previous step. For every image DataGeneratorCustom makes: 

- Cut of an image by patches of 128x128
- Makes three rotations: 90, 180 and 270 degrees
- Shuffles all dataset and yields them one by one with batch_size

3. **Predict with model**. `scripts/predict.py`, for every image it makes following procedure:
- Reads sat image from `dir_test` (default `../data/test_whole_Mecca/sat_z18`)
- Makes upscaling for it because our prediction is always 128x128, for some images the width/height may not divide completely
- Splits every image with sliding window of 128x128 and using parameter `step_size` which is default is 64
- Predicts for array of patches with loaded model from `MODEL_PATH`
- Returns patches back to image meanwhile also normalizing those pixels where there was overlapping for a model
- Cut the image back to initial sizes because as first step we did upscaling
- Writes Heatmap to `output_folder` with geo data inside

4. **To shp files**. `area_assesment/geo/geotiff_to_shp_file.py`. After we made a prediction we need to prepare special file which we will upload to ArcGIS to see the layer there. For this purpose this script reads images from `src_dir` and creates special files for ArcGIS system in `dst_dir`. `m_thre` is parameter of threshold 

5. **Upload to ArcGIS**. Last step is to take `area_assessment\output_data\Mecca\shp` folder, zip it and upload this file to ArcGIS online

### Demo (live): 

Demo of first version:
https://dmitriidenisov.github.io/area_assessment/mecca_demo/area_assesment.html

Demo of second version:
http://35.208.27.233:5555/map prod

http://35.208.27.233:5556/map dev

### Demo (video):
Desktop:
https://youtu.be/dku5B1yJEVY

Mobile:
https://youtu.be/DETdbaptB44

### Deployment:
Models, data, gdal library and area_assessment repo is on `instance-1` server

Frontend application in docker is stored on `front-area-assessment` server

### Statistics and additional information:

May be found in this repository in releases

### Install GDAL library:
Install Gdal library (on Ubuntu): 

```sudo add-apt-repository -y ppa:ubuntugis/ppa
sudo apt update 
sudo apt upgrade # if you already have gdal 1.11 installed 
sudo apt install gdal-bin python-gdal python3-gdal # if you don't have gdal 1.11 already installed
```

Source:
https://stackoverflow.com/questions/37294127/python-gdal-2-1-installation-on-ubuntu-16-04

