# %tensorflow_version 1.x
import tensorflow as tf
import keras
import cv2
import numpy as np
import os

# ---------------------------------- #


INPUT_PATH = 'D:/Documents/Didaktoriko/ACNN/export_data/figure_tiles_32x32/'
_, subCategoryDirectoriesInputSet, _ = next(os.walk(INPUT_PATH))

pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
list_date_ranges = []

#                             input img                           output img
list_tmp = {'carbon': [['path/img_20200101_20200107', 'path/img_20200108_20200114']]}
#
# for country_index in range(0, subCategoryDirectoriesInputSet.__len__()):
#     tmp_path = INPUT_PATH + subCategoryDirectoriesInputSet[country_index]
#     _, list_img_names, _ = next(os.walk(tmp_path))
#     for img_id in range(0, list_img_names.__len__()):
#         tmp_img_name = list_img_names[img_id]

#         for pollution_keyword in pollution_keywords_in_path:
#             if pollution_keyword in tmp_img_name:
#                 for date_range in list_date_ranges:
#                     if date_range in tmp_img_name:

# <Albania>_<carbon_monoxide>_<20200101_20200107>_<size_32x32>_<0>
# <country>_<pollution_key>_<date_range>_<tile_size>_<tile_index>






