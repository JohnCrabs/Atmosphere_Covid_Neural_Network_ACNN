import gdal
import numpy as np
import math
import cv2 as cv

MY_GEO_MIN = 'min'
MY_GEO_MAX = 'max'
MY_GEO_MEAN = 'mean'
MY_GEO_STD_DEV = 'std_dev'
MY_GEO_MEDIAN = 'median'


def mean_calculate(x, y):
    return x / y


def open_image(path: str):
    # print(path)
    return np.nan_to_num(gdal.Open(path).ReadAsArray(), nan=-1.0)


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
        median_px = median_px_list[int(len(median_px_list)/2)]

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
            tmp_export_tiles.append(img[row_index:row_index+tile_size, column_index:column_index+tile_size])

    return tmp_export_tiles


def img_break_tiles_and_export_them(img: [], folder_path, export_name, tile_size=10,  suffix='.png'):
    img *= 255.0 / img.max()
    img_y_size, img_x_size = img.shape
    export_index = 0
    for row_index in range(0, img_y_size, tile_size):
        for column_index in range(0, img_x_size, tile_size):
            cv.imwrite(folder_path + export_name + '_size_' + str(tile_size) + 'x' str(tile_size) + '_'
                                                                                   + str(export_index) + suffix,
                       img[row_index:row_index + tile_size, column_index:column_index + tile_size])
            export_index += 1