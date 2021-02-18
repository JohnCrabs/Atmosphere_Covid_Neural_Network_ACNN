import gdal
import numpy as np
import math
import cv2 as cv
import os

MY_GEO_MIN = 'min'
MY_GEO_MAX = 'max'
MY_GEO_MEAN = 'mean'
MY_GEO_STD_DEV = 'std_dev'
MY_GEO_MEDIAN = 'median'

ZFILL = 4


def mean_calculate(x, y):
    return x / y


def open_geospatial_image_file(path: str):
    # print(path)
    return np.array(gdal.Open(path).ReadAsArray())


def open_image_file(path: str):
    return cv.imread(path)


def export_image_file(path: str, img):
    cv.imwrite(path, img)


def img_statistics(img: [], round_at=3):
    min_px = 0
    max_px = 0
    mean_px = 0
    std_dev = 0
    median_px = 0

    sum_px = 0
    counter_px = 0

    median_px_list = []

    for row in img:
        for px in row:
            if not np.isnan(px):
                if counter_px == 0:
                    min_px = px
                    max_px = px
                else:
                    if px < min_px:
                        min_px = px
                    if px > max_px:
                        max_px = px

                sum_px += px
                counter_px += 1
                median_px_list.append(px)

    if counter_px != 0:
        mean_px = mean_calculate(sum_px, counter_px)
        for x in median_px_list:
            std_dev += (x - median_px) * (x - median_px)
        std_dev = math.sqrt(std_dev / len(median_px_list))

        median_px_list.sort()
        median_px = median_px_list[int(len(median_px_list) / 2)]

    return {MY_GEO_MIN: round(min_px, round_at),
            MY_GEO_MAX: round(max_px, round_at),
            MY_GEO_MEAN: round(mean_px, round_at),
            MY_GEO_STD_DEV: round(std_dev, round_at),
            MY_GEO_MEDIAN: round(median_px, round_at)}


def img_break_tiles(img: [], tile_size=10):
    img *= 255.0 / img.max()
    img_y_size, img_x_size = img.shape
    tmp_export_tiles = []
    for row_index in range(0, img_y_size, tile_size):
        for column_index in range(0, img_x_size, tile_size):
            tmp_export_tiles.append(img[row_index:row_index + tile_size, column_index:column_index + tile_size])

    return tmp_export_tiles


def img_break_tiles_and_export_them(img: [], folder_path, export_name, tile_size=10, suffix='.png',
                                    norm_min_value=None, normalize_max_value=None, info_acc_percent=0.6):
    check_min_value = np.nanmin(img)
    norm_divider = np.nanmax(img)
    if normalize_max_value is not None:
        norm_divider = normalize_max_value
    if norm_min_value is not None:
        check_min_value = norm_min_value

    # img_norm = img * 255.0 / normalize_max_value
    # low_value_flags = img_norm < 0
    # img_norm[low_value_flags] = 0

    img_y_size, img_x_size = img.shape
    tmp_img = [img[row_index:row_index + tile_size, column_index:column_index + tile_size]
               for row_index in range(0, img_y_size, tile_size) for column_index in range(0, img_x_size, tile_size)]
    export_index = 0
    for i in range(len(tmp_img)):
        tmp_x, tmp_y = tmp_img[i].shape
        if tmp_x == tmp_y:
            export_img = np.nan_to_num(tmp_img[i], nan=check_min_value)
            export_img = ((export_img - check_min_value) / (norm_divider - check_min_value))
            non_zero_size = np.count_nonzero(export_img)
            px_size = tmp_x * tmp_y
            info_percent = float(non_zero_size) / float(px_size)

            if info_percent > info_acc_percent:
                export_img = export_img * 255.0 / export_img.max()
                export_image_file(
                    folder_path + export_name + '_size_' + str(tile_size) + 'x' + str(tile_size)
                    + '_tile_' + str(export_index) + '_id_' + suffix, export_img)
                export_index += 1


def find_tiles_number(list_img_paths, list_pollution, list_date_range):
    # find in how many tiles the starting image has been broken
    tile_split_index_size = 0
    for path in list_img_paths:
        if list_pollution[0] in path and list_date_range[0] in path:
            tile_split_index_size += 1
    return tile_split_index_size


def find_tile_paths_matches(list_img_path_dir, list_pollution, list_date_range):
    tmp_img_filenames = os.listdir(list_img_path_dir)
    tile_split_index_size = find_tiles_number(tmp_img_filenames, list_pollution, list_date_range)
    dict_export_tmp = {}
    dict_week_start_end_ranges = {}
    dict_export_index = {}
    for keyword in list_pollution:
        dict_export_tmp[keyword] = []
        dict_week_start_end_ranges[keyword] = []
        dict_export_index[keyword] = []

    for keyword in list_pollution:
        for i in range(0, len(list_date_range) - 1):
            for index in range(tile_split_index_size):
                tmp_list_to_append = []
                tmp_list_date_to_append = []
                break_loop = False
                can_append_list = False
                for filename in tmp_img_filenames:
                    path_to_append = list_img_path_dir + filename
                    if keyword in filename and list_date_range[i] in filename and 'tile_' + \
                            str(index) + '_id' in filename:
                        tmp_list_to_append.append(path_to_append)
                        tmp_list_date_to_append.append(list_date_range[i].split('_')[0])
                        dict_export_index[keyword].append(index)
                        can_append_list = True
                    if keyword in filename and list_date_range[i + 1] in filename and 'tile_' + \
                            str(index) + '_id' in filename:
                        tmp_list_to_append.append(path_to_append)
                        if can_append_list:
                            dict_export_tmp[keyword].append(tmp_list_to_append)
                            tmp_list_date_to_append.append(list_date_range[i+1].split('_')[1])
                            dict_week_start_end_ranges[keyword].append(tmp_list_date_to_append)
                        break_loop = True
                    if break_loop:
                        break

    return dict_export_tmp, dict_week_start_end_ranges, dict_export_index
