import datetime as dt
import os  # import os
import time  # import time

import matplotlib.pyplot as plt  # import pyplot for figure plotting
import numpy as np
import pandas as pd  # import pandas for array processing

import my_lib.my_geo_img as my_geo_img  # import my_geo_img lib (for image processing)
import test_uNet as my_uNet
from google_lib import my_gee  # import my_gee lib (for satellite images)
from my_lib import my_calendar_v2 as my_cal_v2  # import my calendar lib (for date anaysis)

import shutil

pd.options.mode.chained_assignment = None  # this line prevents an error of pd.fillna() function

# ----------------------------------------- #
# ---------- 0) Define Variables ---------- #
# ----------------------------------------- #
_ON = True  # Create a switch ON (use this for better visualization of the code)
_OFF = False  # Create a switch OFF (use this for better visualization of the code)

S05_SCALE_M_PER_PX = 5000
S05_SLEEP_MINUTE_MULTIPLIER = 0.0
S05_SLEEP_TIME_IN_SEC = int(S05_SLEEP_MINUTE_MULTIPLIER * 60.0)

_TILE_SIZE = 32  # the width height for the country tiles
_EPOCHS = 5  # the epochs for running the uNet

_START_RESEARCH_DATE = "2020-01-01"  # "YYYY-MM-DD"
_END_RESEARCH_DATE = "2020-12-31"  # "YYYY-MM-DD"

# Create the period step
# a multiple of week period (by week I dont mean Monday to Sunday necessary, but a 7 day in a row)
_PERIOD_ALPHA_MULTIPLIER = 1  # a multiplier for weeks
_PERIOD_STEP = _PERIOD_ALPHA_MULTIPLIER * 6  # the period step

_FLAG_FOR_PLOT_COVID_COUNTRY_FIGURES = _OFF  # a flag for time saving purposes
_FLAG_FOR_DOWNLOADING_SENTINEL_DATA = _OFF  # a flag for time saving purposes
_FLAG_FOR_BREAKING_SENTINEL_IMAGES_TO_TILES = _OFF  # a flag for time saving purposes
_FLAG_FOR_FINDING_DTILES = _OFF  # a flag for time saving purposes
_FLAG_FOR_CREATING_COVID_IMAGES = _OFF  # a flag for time saving purposes
_FLAG_FOR_TRAINING_UNET = _OFF  # a flag for time saving purposes
_FLAG_FOR_TESTING_UNET = _OFF  # a flag for time saving purposes

# FOLDER PATHS
_PATH_PRIMARY_DIR = 'D:/Documents/Didaktoriko/ACNN/'
_PATH_FOR_FIGURE_PLOTS = _PATH_PRIMARY_DIR + 'Figure_Plots/'
_PATH_FOR_SAT_IMAGE_TILES = _PATH_PRIMARY_DIR + 'Figure_Tiles/' + str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) + '/'
_PATH_FOR_DTILES = _PATH_PRIMARY_DIR + 'Figure_Dtiles/' + str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) + '/'
_PATH_FOR_COVID_MEASURE_TILES = _PATH_PRIMARY_DIR + 'Covid_Measures/' + str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) + '/'
_PATH_FOR_TRAIN_VAL_TEST_INDEXES = _PATH_PRIMARY_DIR + 'Train_Val_Test_' + str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) + '/'
_PATH_FOR_NN_MODEL = _PATH_PRIMARY_DIR + 'NN_Models/'
_PATH_FOR_NN_MODEL_NAME = _PATH_PRIMARY_DIR + 'NN_Model_Names/'

_PATH_FOR_COVID_CSV_DIR = _PATH_PRIMARY_DIR + 'Data_Files/Covid_CSV/'


# ----------------------------------------- #
# ---------- 1) Define Functions ---------- #
# ----------------------------------------- #


def check_if_file_exists(path):
    if os.path.exists(path) and os.path.isfile(path):
        return True
    else:
        print("File cannot be found or doesn't exist: " + path)
        return False


def check_create_folders_in_path(path):
    """
    Create the corresponding directory path.
    :param path: The path to be created
    :return:
    """
    if not os.path.exists(path):  # if path does not exist
        os.makedirs(path)  # create path
        print('Path succesfully created: ' + path)  # print message
    else:  # print message
        print('Path already exists: ' + path)  # print message


def clear_dir_files(path):
    """
    Delete all files and folders in a specified directory path.
    :param path: directory path for file deletion
    :return:
    """
    print()
    if os.path.exists(path):  # check if path exists
        list_dir = os.listdir(path)  # list all directories in path
        for dir_ in list_dir:  # for each directory in path
            print("Removing " + path + dir_)  # print message
            shutil.rmtree(path + dir_)  # delete directories content


def find_unique_values_list(list_input: []):
    """
    Find the unique values in a list
    :param list_input:
    :return:
    """
    list_output = []  # create a temporary list output
    for item in list_input:  # for each item in list input
        if item not in list_output:  # if item is not in list output
            list_output.append(item)  # append it to list_output
    return list_output  # return the list output


def create_dictionary_from_list_column(list_input: [], key_column_index: int):
    """
    create a dictionary from the list collumns
    :param list_input:
    :param key_column_index:
    :return:
    """
    dict_output = {}  # create a dictionary for output
    for row in list_input:  # for each row in list_input
        if row[key_column_index] not in dict_output.keys():  # if value is not in dictionary (prevents destroy the list)
            dict_output[row[key_column_index]] = []  # create a new dictionary with empty key
        tmp_list = []  # create a temporary list (for append)
        for func_index in range(0, len(row)):  # for func_index in range()
            if func_index != key_column_index:  # if func_index is different than key_column_index (exclude this value)
                tmp_list.append(row[func_index])  # append value to tmp_list
        dict_output[row[key_column_index]].append(tmp_list)  # append the list to new list
    return dict_output  # return the dictionary


# --------------------------------------------- #
# ---------- 2) Check Path Existence ---------- #
# --------------------------------------------- #

check_create_folders_in_path(_PATH_PRIMARY_DIR)
check_create_folders_in_path(_PATH_FOR_FIGURE_PLOTS)
check_create_folders_in_path(_PATH_FOR_SAT_IMAGE_TILES)
check_create_folders_in_path(_PATH_FOR_DTILES)
check_create_folders_in_path(_PATH_FOR_COVID_MEASURE_TILES)
check_create_folders_in_path(_PATH_FOR_TRAIN_VAL_TEST_INDEXES)
check_create_folders_in_path(_PATH_FOR_NN_MODEL)
check_create_folders_in_path(_PATH_FOR_NN_MODEL_NAME)
check_create_folders_in_path(_PATH_FOR_COVID_CSV_DIR)

# ------------------------------------- #
# ---------- 3) Create Lists ---------- #
# ------------------------------------- #

# Break the date range into smaller ranges
_LIST_DATE_RANGES_PERIOD = my_cal_v2.break_date_range_to_periods(date_start=_START_RESEARCH_DATE,
                                                                 date_end=_END_RESEARCH_DATE,
                                                                 period_step=_PERIOD_STEP,
                                                                 date_format=my_cal_v2.YYYY_MM_DD,
                                                                 date_delimeter=my_cal_v2.del_dash, century=21)

# Create a list with stings of the date range.
# e.g. ['2020-01-01', '2020-01-07'] => '20200101_20200107'
_LIST_DATE_RANGES_FOR_PATHS = my_cal_v2.create_string_list_date_range(list_input=_LIST_DATE_RANGES_PERIOD,
                                                                      del_input=my_cal_v2.del_dash,
                                                                      del_output=my_cal_v2.del_none)

# print(list_date_period_ranges)
# print(list_date_range_path_string)

# Sentinel Lists
_LIST_SENTINEL_COLLECTION_IDS = ['COPERNICUS/S5P/OFFL/L3_O3', 'COPERNICUS/S5P/OFFL/L3_CO',
                                 'COPERNICUS/S5P/OFFL/L3_AER_AI', 'COPERNICUS/S5P/OFFL/L3_HCHO',
                                 'COPERNICUS/S5P/OFFL/L3_SO2', 'COPERNICUS/S5P/OFFL/L3_NO2']

_LIST_SENTINEL_COLLECTION_BANDS = ['O3_column_number_density', 'CO_column_number_density', 'absorbing_aerosol_index',
                                   'tropospheric_HCHO_column_number_density', 'SO2_column_number_density',
                                   'NO2_column_number_density']

_LIST_SENTINEL_COLLECTION_NAMES = ['ozone_O3_density', 'carbon_monoxide_CO_density', 'absorbing_aerosol_index',
                                   'offline_formaldehyde_HCHO_density', 'sulphur_dioxide_SO2_density',
                                   'nitrogen_dioxide_NO2_density']

# Create a list of headers for (used later for the final array)
_LIST_COVID_MEASURES_HEADERS_FOR_POLLUTION = ['DATE_RANGE', 'RESTRICTIONS_INTERNAL_MOVEMENTS',
                                              'INTERNATIONAL_TRAVEL_CONTROLS', 'CANCEL_PUBLIC_EVENTS',
                                              'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

# df_x_plot and df_y_plot used to export plots (for data visualization)
_STR_X_AXIS_PLOT = _LIST_COVID_MEASURES_HEADERS_FOR_POLLUTION[0]
_LIST_Y_AXIS_PLOT = _LIST_COVID_MEASURES_HEADERS_FOR_POLLUTION[1:]

# ------------------------------------------------- #
# ---------- 4) Create Datasets From CSV ---------- #
# ------------------------------------------------- #


'''
# ############################
# ### DATASET COLUMN NAMES ###
# ############################
# COUNTRY: Covid, Pollution               ID: Covid, Pollution                       DATE: Covid Pollution
# RETAIL_AND_RECREATIONAL: None           GROCERY_AND_PHARMACY: None                 PARKS,TRANSIT_STATION: None
# WORKPLACES,RESIDENTIAL: None            TESTING_POLICY: None                       CONTACT_TRACING: Covid
# CONTAINMENT_INDEX: None                 STRINGENCY_INDEX: None                     VACCINATION_POLICY: Covid
# DEPT_RELIEF: None                       FACIAL_COVERING: None                      INCOME_SUPPORT: None
# RESTRICTIONS_INTERNAL_MOVEMENTS: Covid, Pollution   
# INTERNATIONAL_TRAVEL_CONTROLS: Covid, Pollution
# CONTINENT: Covid, Pollution             TOTAL_CASES: Covid                         NEW_CASES: Covid
# NEW_CASES_SMOOTHED: Covid               TOTAL_DEATHS: Covid                        NEW_DEATHS: Covid
# NEW_DEATHS_SMOOTHED: Covid              TOTAL_CASES_PER_MILLION: Covid             NEW_CASES_PER_MILLION: Covid
# NEW_CASES_SMOOTHED_PER_MILLION: Covid   TOTAL_DEATHS_PER_MILLION: Covid            NEW_DEATHS_PER_MILLION: Covid
# NEW_DEATHS_SMOOTHED_PER_MILLION: Covid  REPRODUCTION_RATE: Covid                   ICU_PATIENTS: Covid
# ICU_PATIENTS_PER_MILLION: Covid         HOSP_PATIENTS: Covid                       HOSP_PATIENTS_PER_MILLION: Covid
# WEEKLY_ICU_ADMISSIONS: Covid            WEEKLY_ICU_ADMISSIONS_PER_MILLION: Covid   WEEKLY_HOSP_ADMISSIONS: Covid
# WEEKLY_HOSP_ADMISSIONS_PER_MILLION: Covid
# NEW_TEST: Covid                         TOTAL_TESTS: Covid                         TOTAL_TESTS_PER_THOUSAND: Covid
# NEW_TESTS_PER_THOUSAND: Covid           NEW_TESTS_SMOOTHED: Covid
# NEW_TESTS_SMOOTHED_PER_THOUSAND: Covid  POSITIVE_RATE: Covid                       TESTS_PER_CASE: Covid
# TESTS_UNITS: Covid                      TOTAL_VACCINATIONS: Covid                  NEW_VACCINATIONS: Covid
# TOTAL_VACCINATIONS_PER_HUNDRED: Covid   NEW_VACCINATIONS_PER_MILLION: Covid        POPULATION: Covid, Pollution
# POPULATION_DENSITY: Covid, Pollution    MEDIAN_AGE: Covid                          AGED_65_OLDER: Covid
# AGED_70_OLDER: Covid                    GDP_PER_CAPITA: Covid                      EXTREME_POVERTY: Covid
# CARDIOVASC_DEATH_RATE: Covid            DIABETES_PREVALENCE: Covid                 FEMALE_SMOKERS: Covid
# MALE_SMOKERS: Covid                     HANDWASHING_FACILITIES: Covid              HOSPITAL_BEDS_PER_THOUSAND: Covid
# LIFE_EXPECTANCY: Covid                  HUMAN_DEVELOPMENT_INDEX: None              PUBLIC_INFORMATION_CAMPAIGNS: Covid
# CANCEL_PUBLIC_EVENTS: Covid, Pollution
# RESTRICTION_GATHERINGS: Covid, Pollution
# CLOSE_PUBLIC_TRANSPORT: Covid, Pollution
# SCHOOL_CLOSURES: Covid, Pollution
# STAY_HOME_REQUIREMENTS: Covid, Pollution
# WORKPLACE_CLOSURES: Covid, Pollution
'''

path_for_covid_measures_csv_file = _PATH_FOR_COVID_CSV_DIR + 'covid_measures.csv'
df_covid_measures = pd.DataFrame()
if check_if_file_exists(path_for_covid_measures_csv_file):
    df_covid_measures = pd.read_csv(path_for_covid_measures_csv_file, low_memory=False)  # read the csv file
else:
    exit(404)

_LIST_UNIQUE_COUNTRIES = df_covid_measures.COUNTRY.unique()  # create a list with the countries included in list


# print(list_unique_countries)
# print(len(list_unique_countries))


# Create the Dataframe to be used for training the pollution Neural Network
df_pollution_dataset = df_covid_measures[['COUNTRY', 'DATE', 'RESTRICTIONS_INTERNAL_MOVEMENTS',
                                          'INTERNATIONAL_TRAVEL_CONTROLS', 'CANCEL_PUBLIC_EVENTS',
                                          'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                                          'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']]

# Create the Dataframe to be used for training the covid Neural Network
df_covid_dataset = df_covid_measures[['COUNTRY', 'ID', 'CONTACT_TRACING', 'VACCINATION_POLICY',
                                      'RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                                      'CONTINENT', 'TOTAL_CASES', 'NEW_CASES_SMOOTHED', 'TOTAL_DEATHS',
                                      'NEW_CASES', 'NEW_CASES_SMOOTHED', 'TOTAL_CASES_PER_MILLION',
                                      'NEW_CASES_PER_MILLION', 'NEW_CASES_SMOOTHED_PER_MILLION', 'REPRODUCTION_RATE',
                                      'ICU_PATIENTS', 'ICU_PATIENTS_PER_MILLION', 'HOSP_PATIENTS',
                                      'HOSP_PATIENTS_PER_MILLION', 'WEEKLY_ICU_ADMISSIONS',
                                      'WEEKLY_ICU_ADMISSIONS_PER_MILLION', 'WEEKLY_HOSP_ADMISSIONS',
                                      'WEEKLY_HOSP_ADMISSIONS_PER_MILLION', 'NEW_TESTS', 'TOTAL_TESTS',
                                      'TOTAL_TESTS_PER_THOUSAND', 'NEW_TESTS_PER_THOUSAND', 'NEW_TESTS_SMOOTHED',
                                      'NEW_TESTS_SMOOTHED_PER_THOUSAND', 'POSITIVE_RATE', 'TESTS_PER_CASE',
                                      'TESTS_UNITS', 'TOTAL_VACCINATIONS', 'NEW_VACCINATIONS',
                                      'TOTAL_VACCINATIONS_PER_HUNDRED', 'NEW_VACCINATIONS_PER_MILLION', 'POPULATION',
                                      'POPULATION_DENSITY', 'MEDIAN_AGE', 'AGED_65_OLDER', 'AGED_70_OLDER',
                                      'GDP_PER_CAPITA', 'EXTREME_POVERTY', 'CARDIOVASC_DEATH_RATE',
                                      'DIABETES_PREVALENCE', 'FEMALE_SMOKERS', 'MALE_SMOKERS',
                                      'HANDWASHING_FACILITIES', 'HOSPITAL_BEDS_PER_THOUSAND', 'LIFE_EXPECTANCY',
                                      'HUMAN_DEVELOPMENT_INDEX', 'PUBLIC_INFORMATION_CAMPAIGNS', 'CANCEL_PUBLIC_EVENTS',
                                      'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT', 'SCHOOL_CLOSURES',
                                      'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']]

# fillna
df_pollution_dataset.fillna(method='ffill', inplace=True)
# create a dictionary from the list (using the countries as a key column)
df_polllution_dict = create_dictionary_from_list_column(list_input=df_pollution_dataset.values.tolist(),
                                                        key_column_index=0)


# --------------------------------------------------- #
# ---------- 5) Download Sentinel-5 Images ---------- #
# --------------------------------------------------- #

# my_gee.clear_tasks() # uncomment this line to stop all tasks in Google Earth Engine
if _FLAG_FOR_DOWNLOADING_SENTINEL_DATA:
    # list_S5_data_len = len(_LIST_SENTINEL_COLLECTION_IDS)  # take the length of
    # for i in range(0, list_S5_data_len):
    #     my_gee.download_image_from_collection(collection_id=list_collection_id[i],
    #                                           image_band=list_colection_bands[i],
    #                                           img_name=list_collection_names[i],
    #                                           list_date_range=list_date_period_ranges,
    #                                           list_countries=list_unique_countries,
    #                                           scale=scale_m_per_px,
    #                                           waiting_time=waiting_time_in_sec)
    my_gee.download_image_from_collection(collection_id=_LIST_SENTINEL_COLLECTION_IDS[5],
                                          image_band=_LIST_SENTINEL_COLLECTION_BANDS[5],
                                          img_name=_LIST_SENTINEL_COLLECTION_NAMES[5],
                                          list_date_range=_LIST_DATE_RANGES_PERIOD,
                                          list_countries=_LIST_UNIQUE_COUNTRIES,
                                          scale=S05_SCALE_M_PER_PX,
                                          waiting_time=S05_SLEEP_TIME_IN_SEC)

# -------------------------------------------- #
# ---------- 5) PLOT COVID MEASURES ---------- #
# -------------------------------------------- #
# print(df_polllution_dict.keys())
df_pollution_mean_range_dict = {}  #
df_pollution_mean_range_dict_with_headers = {}
if _FLAG_FOR_PLOT_COVID_COUNTRY_FIGURES:  # If True
    for key in df_polllution_dict.keys():  # for each key in pollution df dictionary
        df_pollution_mean_range_dict[key.replace(' ', '_')] = []  # add key as a list
        df_pollution_mean_range_dict[key.replace(' ', '_')] = my_cal_v2.merge_values_in_date_range_list(
            list_input=df_polllution_dict.copy()[key],
            date_index=0,
            date_range_list=_LIST_DATE_RANGES_PERIOD,
            merge_type=my_cal_v2.merge_mean,
            del_input=my_cal_v2.del_dash,
            del_output=my_cal_v2.del_none,
            del_use=True)  # merge values using date format

        # print(key, df_pollution_mean_range_dict[key])

    # Find statistics and plot them
    for key in df_pollution_mean_range_dict.keys():  # for each key in newly created dict
        path_to_export_plot_png = _PATH_FOR_FIGURE_PLOTS + key + '/'  # create new dir
        check_create_folders_in_path(path_to_export_plot_png)  # check if path exist and create it otherwise
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')] = []  # create empty list

        for dataset_index in range(0, len(df_pollution_mean_range_dict[key])):  # for each dataset (take index)
            tmp_dataset_list = [df_pollution_mean_range_dict[key][dataset_index][0]]  # first list is the name
            for value in df_pollution_mean_range_dict[key][dataset_index][1]:  # for each value
                tmp_dataset_list.append(value)  # append the value
            df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')].append(tmp_dataset_list)  # append it to dict
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')] = pd.DataFrame(
            df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')],
            columns=_LIST_COVID_MEASURES_HEADERS_FOR_POLLUTION)  # make it dataFrame

        # Plot the values
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')].plot(x=_STR_X_AXIS_PLOT, y=_LIST_Y_AXIS_PLOT)
        plt.gcf().set_size_inches(20.48, 10.24)
        plt.xticks(rotation=90)
        plt.xlim(0, 53)
        plt.ylim(0, 6)
        plt.title(key)
        plt.xlabel('Date_Range')
        plt.ylabel('Covid Measures')
        plt.xticks(range(0, 52))
        plt.savefig(path_to_export_plot_png + key.replace(' ', '_') + "_covid_measures.png", dpi=100)
        time.sleep(1)
        print("Plot " + key + " exported")


# # ----------------------------------------------- #
# # ---------- 4a) Read Sentiner-5 Images ---------- #
# # ----------------------------------------------- #
#
# df_satelite_image_data_dict = {}
# tmp_test_country = ['Greece']
# pollution_statistic_header = ['DATE_RANGE', 'MIN', 'MAX', 'MEAN', 'STD_DEV', 'MEDIAN']
# pollution_keywords_in_path = ['carbon_monoxide', 'ozone', "sulphur_dioxide", "nitrogen_dioxide"]
# if print_covid_country_plot_fig:
#     img_path_folder = "Data/Satellite_Atmospheric_Images/tiff_folders/GEE_"
#
#     for country in list_unique_countries_for_path:
#         # for country in tmp_test_country:
#         tmp_img_path_folder = img_path_folder + country + "/"
#         img_path_files = os.listdir(tmp_img_path_folder)
#
#         path_to_export_plot_png = path_for_saving_plots + country + '/'
#         if not os.path.exists(path_to_export_plot_png):
#             os.mkdir(path_to_export_plot_png)
#
#         dict_pollution_stats = {}
#         for keyword in pollution_keywords_in_path:
#             dict_pollution_stats[keyword] = []
#             for date_range in list_date_range_path_string:
#                 for path_file in img_path_files:
#                     if keyword in path_file and date_range in path_file:
#                         img = my_geo_img.open_image(tmp_img_path_folder + path_file)
#                         img_stats = my_geo_img.img_statistics(img, round_at=5)
#
#                         tmp_stats_list = [date_range,
#                                           img_stats[my_geo_img.MY_GEO_MIN],
#                                           img_stats[my_geo_img.MY_GEO_MAX],
#                                           img_stats[my_geo_img.MY_GEO_MEAN],
#                                           img_stats[my_geo_img.MY_GEO_STD_DEV],
#                                           img_stats[my_geo_img.MY_GEO_MEDIAN]]
#
#                         dict_pollution_stats[keyword].append(tmp_stats_list)
#             df_pollution_stats = pd.DataFrame(dict_pollution_stats[keyword], columns=pollution_statistic_header)
#             df_pollution_stats.plot(x=df_x_plot, y=['MIN', 'MAX', 'MEAN', 'STD_DEV', 'MEDIAN'])
#             plt.gcf().set_size_inches(20.48, 10.24)
#             plt.xticks(rotation=90)
#             plt.xlim(0, 53)
#             # plt.ylim(0, 6)
#             plt.title(country + "_" + keyword + "_" + "density")
#             plt.xlabel('Date_Range')
#             plt.ylabel('Density')
#             plt.xticks(range(0, 53))
#             plt.savefig(path_to_export_plot_png + country + "_" + keyword + ".png", dpi=100)
#             time.sleep(1)
#             print("Plot " + country + " " + keyword + " exported")
#             df_satelite_image_data_dict[country] = dict_pollution_stats.copy()
#
# # print(dict_pollution_stats)
#
# # img = geo_img.open_image(img_path)
# # img_stats = geo_img.img_statistics(img, round_at=5)
# # print(img_stats)
#
# # --------------------------------------------- #
# # ---------- 5a) Create Exporting CSV ---------- #
# # --------------------------------------------- #
#
# print(df_pollution_mean_range_dict)
# print(df_satelite_image_data_dict)
# csv_output_list = [['COUNTRY', 'DATE_RANGE', 'RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
#                     'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT', 'SCHOOL_CLOSURES',
#                     'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES', 'CARBON_MONOOXIDE_MAX', 'CARBON_MONOOXIDE_MIN',
#                     'CARBON_MONOOXIDE_MEAN', 'CARBON_MONOOXIDE_STDDEV', 'CARBON_MONOOXIDE_MEDIAN', 'OZONE_MAX',
#                     'OZONE_MIN', 'OZONE_MEAN', 'OZONE_STDDEV', 'OZONE_MEDIAN', 'SULPHUR_DIOXIDE_MAX',
#                     'SULPHUR_DIOXIDE_MIN', 'SULPHUR_DIOXIDE_MEAN', 'SULPHUR_DIOXIDE_STDDEV', 'SULPHUR_DIOXIDE_MEDIAN',
#                     'NITROGEN_DIOXIDE_MAX', 'NITROGEN_DIOXIDE_MIN', 'NITROGEN_DIOXIDE_MEAN', 'NITROGEN_DIOXIDE_STDDEV',
#                     'NITROGEN_DIOXIDE_MEDIAN']]
#
# for country in list_unique_countries_for_path:
#     # for country in tmp_test_country:
#     for date_range in list_date_range_path_string:
#         tmp_append_list = [country, date_range]
#         for row in df_pollution_mean_range_dict[country]:
#             if row[0] == date_range:
#                 for value in row[1]:
#                     tmp_append_list.append(value)
#         for key in pollution_keywords_in_path:
#             for row in df_satelite_image_data_dict[country][key]:
#                 if date_range in row:
#                     for index in range(1, len(row)):
#                         tmp_append_list.append(row[index])
#         csv_output_list.append(tmp_append_list)
#
# csv_path = "export_data/acnn_data.csv"
# my_cal_v2.write_csv(csv_path=csv_path, list_write=csv_output_list, delimeter=my_cal_v2.del_comma)

# ------------------------------------------------------- #
# ---------- 4b) Break Satelite Images to Tiles ---------- #
# ------------------------------------------------------- #

if _FLAG_FOR_BREAKING_SENTINEL_IMAGES_TO_TILES:
    clear_dir_files(_PATH_FOR_SAT_IMAGE_TILES)
    print()
    img_path_folder = "Data/Satellite_Atmospheric_Images/tiff_folders/GEE_"
    # pollution_keywords_in_path = ['carbon_monoxide', 'ozone', "sulphur_dioxide", "nitrogen_dioxide"]
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    pollution_dict_min_max_values = {'carbon_monoxide': {'min': 0.00, 'max': 0.10},  # mol/m^2
                                     'ozone': {'min': 0.10, 'max': 0.18}}  # mol/m^2
    country_index = 1
    for country in _LIST_UNIQUE_COUNTRIES:
        print('(' + str(country_index) + ' / ' + str(len(_LIST_UNIQUE_COUNTRIES))
              + ') Exporting tiles for ' + country.replace(' ', '_'))
        country_index += 1
        tmp_img_path_folder = img_path_folder + country.replace(' ', '_') + "/"
        img_path_files = os.listdir(tmp_img_path_folder)
        path_to_export_tile_png = _PATH_FOR_SAT_IMAGE_TILES + country.replace(' ', '_') + '/'
        if not os.path.exists(path_to_export_tile_png):
            os.mkdir(path_to_export_tile_png)
        for keyword in pollution_keywords_in_path:
            for date_range in _LIST_DATE_RANGES_FOR_PATHS:
                for path_file in img_path_files:
                    if keyword in path_file and date_range in path_file:
                        # print(tmp_img_path_folder + path_file)
                        img = my_geo_img.open_geospatial_image_file(tmp_img_path_folder + path_file)
                        my_geo_img.img_break_tiles_and_export_them(img=img,
                                                                   tile_size=_TILE_SIZE,
                                                                   folder_path=path_to_export_tile_png,
                                                                   export_name=country.replace(' ',
                                                                                               '_') + '_' + keyword + '_' + date_range,
                                                                   norm_min_value=
                                                                   pollution_dict_min_max_values[keyword]['min'],
                                                                   normalize_max_value=
                                                                   pollution_dict_min_max_values[keyword]['max'],
                                                                   info_acc_percent=0.6)

# --------------------------------------------------------- #
# ---------- 5b) Break Satelite Images to DTiles ---------- #
# --------------------------------------------------------- #

if _FLAG_FOR_FINDING_DTILES:
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    import_img_path_folder = _PATH_FOR_SAT_IMAGE_TILES
    clear_dir_files(_PATH_FOR_DTILES)
    print()
    for country in _LIST_UNIQUE_COUNTRIES:
        # print()
        print('Exporting Dtiles for ' + country)
        export_dir_path = _PATH_FOR_DTILES + country.replace(' ', '_') + '/'
        if not os.path.exists(export_dir_path):
            os.makedirs(export_dir_path)

        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=_LIST_DATE_RANGES_FOR_PATHS)
        for key in dict_with_tile_paths.keys():
            for i in range(0, len(dict_with_tile_paths[key])):
                d_date_range_path = dict_with_tile_paths[key][i]
                d_date_range = dict_week_start_end_ranges[key][i]
                export_index = dict_export_index[key][i]

                # Uncomment for debugging
                # print()
                # print(dict_with_dtile_paths[key][i])
                # print(dict_week_start_end_ranges[key][i])
                # print(dict_export_index[key][i])

                img_now = my_geo_img.open_image_file(d_date_range_path[0])
                img_next = my_geo_img.open_image_file(d_date_range_path[1])
                dtile_img = img_next - img_now
                export_img_path = (
                        export_dir_path + country.replace(' ', '_') + '_' + key + '_' + d_date_range[0] + '_'
                        + d_date_range[1] + '_size_' + str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) +
                        '_tile_' + str(export_index) + '_id' + '.png')

                my_geo_img.export_image_file(export_img_path, dtile_img)

# ----------------------------------------------- #
# ---------- 5c) Train Unet with Tiles ---------- #
# ----------------------------------------------- #
if _FLAG_FOR_CREATING_COVID_IMAGES:
    def create_measure_img(measure_value, width_size, height_size):
        tmp_list = []
        for _ in range(width_size):
            tmp_wid = []
            for __ in range(height_size):
                tmp_wid.append(measure_value)
            tmp_list.append(tmp_wid)
        return tmp_list


    path_acnn_data = 'Data/acnn_data.csv'
    df_acnn = pd.read_csv(path_acnn_data, low_memory=False)

    df_dict_acnn = df_acnn[['COUNTRY',
                            'DATE_RANGE',
                            'RESTRICTIONS_INTERNAL_MOVEMENTS',
                            'INTERNATIONAL_TRAVEL_CONTROLS',
                            'CANCEL_PUBLIC_EVENTS',
                            'RESTRICTION_GATHERINGS',
                            'CLOSE_PUBLIC_TRANSPORT',
                            'SCHOOL_CLOSURES',
                            'STAY_HOME_REQUIREMENTS',
                            'WORKPLACE_CLOSURES']]

    df_acnn = df_acnn[['COUNTRY',
                       'DATE_RANGE',
                       'RESTRICTIONS_INTERNAL_MOVEMENTS',
                       'INTERNATIONAL_TRAVEL_CONTROLS',
                       'CANCEL_PUBLIC_EVENTS',
                       'RESTRICTION_GATHERINGS',
                       'CLOSE_PUBLIC_TRANSPORT',
                       'SCHOOL_CLOSURES',
                       'STAY_HOME_REQUIREMENTS',
                       'WORKPLACE_CLOSURES']].values.tolist()

    covid_measures_headers = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                              'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

    clear_dir_files(_PATH_FOR_COVID_MEASURE_TILES)
    print()
    for country in _LIST_UNIQUE_COUNTRIES:
        print('Export covid measure images for ' + country)
        export_id = 0
        check_create_folders_in_path(_PATH_FOR_COVID_MEASURE_TILES + country.replace(' ', '_') + '/')
        for date_range in _LIST_DATE_RANGES_FOR_PATHS:
            for measure in covid_measures_headers:
                for i in range(0, len(df_acnn)):
                    if country.replace(' ', '_') == df_acnn[i][0] and date_range == df_acnn[i][1]:
                        norm_value = (df_dict_acnn[measure][i] / 4.0) * 255
                        tmp_measure_img = create_measure_img(norm_value, _TILE_SIZE, _TILE_SIZE)
                        img_export_path = (
                                _PATH_FOR_COVID_MEASURE_TILES + country.replace(' ', '_') + '/' +
                                country.replace(' ', '_') + '_' + measure + '_' + date_range + '_' + 'size_' +
                                str(_TILE_SIZE) + 'x' + str(_TILE_SIZE) + '_tile_' + str(export_id) + '_id' + '.png')
                        my_geo_img.export_image_file(img_export_path, np.array(tmp_measure_img))
                        break

# ----------------------------------------------- #
# ---------- 6a) Train Unet with Tiles ---------- #
# ----------------------------------------------- #
if _FLAG_FOR_TRAINING_UNET:
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    covid_measures_headers = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                              'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

    epochs = _EPOCHS
    nn_uNet = my_uNet.MyUNet()
    nn_uNet.set_uNet(height=_TILE_SIZE, width=_TILE_SIZE, channels_input=10, channels_output=2, n_filters=16)

    import_img_path_folder = _PATH_FOR_SAT_IMAGE_TILES
    import_img_path_covid_folder = _PATH_FOR_COVID_MEASURE_TILES
    # country = 'Albania'
    tmp_results_index = 0
    export_train_val_test_data_path_with_epochs = _PATH_FOR_TRAIN_VAL_TEST_INDEXES + 'epochs_' + str(epochs).zfill(
        4) + '/'
    check_create_folders_in_path(export_train_val_test_data_path_with_epochs)
    clear_dir_files(export_train_val_test_data_path_with_epochs)
    start_time = dt.datetime.now()
    for country in _LIST_UNIQUE_COUNTRIES:
        print('\n\n\n')
        print('Train uNet for ' + country + ' ' + dt.datetime.now().strftime("(%Y-%m-%d %H:%M:%S)"))
        print('Process started at: ', start_time.strftime("(%Y-%m-%d %H:%M:%S)"))
        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=_LIST_DATE_RANGES_FOR_PATHS)

        dict_with_covid_paths, dict_covid_week_start_end_ranges, dict_covid_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_covid_folder + country.replace(' ', '_') + '/',
            list_pollution=covid_measures_headers,
            list_date_range=_LIST_DATE_RANGES_FOR_PATHS)

        df_xs = []
        df_ys = []
        # print(len(dict_with_dtile_paths[pollution_keywords_in_path[0]]))
        # print(len(dict_covid_export_index[covid_measures_headers[0]]))
        min_keywords_len = None
        for keyword in pollution_keywords_in_path:
            if min_keywords_len is None:
                min_keywords_len = len(dict_with_tile_paths[keyword])
            else:
                if len(dict_with_tile_paths[keyword]) < min_keywords_len:
                    min_keywords_len = len(dict_with_tile_paths[keyword])

        min_covid_keywords_len = None
        for keyword in covid_measures_headers:
            if min_covid_keywords_len is None:
                min_covid_keywords_len = len(dict_with_covid_paths[keyword])
            else:
                if len(dict_with_covid_paths[keyword]) < min_keywords_len:
                    min_covid_keywords_len = len(dict_with_covid_paths[keyword])

        if min_keywords_len != 0 and min_covid_keywords_len != 0:
            for paths_id in range(0, min_keywords_len):
                tmp_xs = []
                tmp_ys = []
                for keyword in pollution_keywords_in_path:
                    if keyword is pollution_keywords_in_path[0]:
                        for covid_keyword in covid_measures_headers:
                            for covid_id in range(min_covid_keywords_len):
                                if dict_week_start_end_ranges[keyword][paths_id] == \
                                        dict_covid_week_start_end_ranges[covid_keyword][covid_id]:
                                    # print(dict_week_start_end_ranges[keyword][paths_id], dict_covid_week_start_end_ranges[covid_keyword][covid_id])
                                    x_img = my_geo_img.open_image_file(
                                        dict_with_covid_paths[covid_keyword][covid_id][0])
                                    tmp_xs.append(x_img)
                                    break

                    x_img = my_geo_img.open_image_file(dict_with_tile_paths[keyword][paths_id][0])  # this week
                    y_img = my_geo_img.open_image_file(dict_with_tile_paths[keyword][paths_id][1])  # next week
                    tmp_xs.append(x_img)
                    tmp_ys.append(y_img)
                tmp_xs = np.array(tmp_xs).T
                tmp_ys = np.array(tmp_ys).T
                df_xs.append(tmp_xs)
                df_ys.append(tmp_ys)

            df_xs = np.array(df_xs) / 255.0
            df_ys = np.array(df_ys) / 255.0

            # print(df_xs.shape)
            # print(df_ys.shape)

            # TRAIN - TEST - VALIDATION INDEXES
            indexes = np.random.permutation(df_xs.shape[0])
            train_percentage = 0.70
            valid_percentage = 0.10
            test_percentage = 1 - (train_percentage + valid_percentage)
            where_to_cut = [int(len(indexes) * train_percentage),
                            int(len(indexes) * (train_percentage + valid_percentage))]
            train_indexes = indexes[0:where_to_cut[0]]
            valid_indexes = indexes[where_to_cut[0] + 1:where_to_cut[1]]
            test_indexes = indexes[where_to_cut[1] + 1:]

            df_dict_train_val_test = [pd.DataFrame.from_dict({'train_indexes': train_indexes}),
                                      pd.DataFrame.from_dict({'val_indexes': valid_indexes}),
                                      pd.DataFrame.from_dict({'test_indexes': test_indexes})]

            export_train_val_test_dir_path = export_train_val_test_data_path_with_epochs + country + '/'
            check_create_folders_in_path(export_train_val_test_dir_path)
            for df_array in df_dict_train_val_test:
                export_train_val_test_file_path = export_train_val_test_dir_path + country + '_' + df_array.keys()[
                    0] + '.csv'
                df_array.to_csv(export_train_val_test_file_path)

            results = nn_uNet.train_uNet(X_train=df_xs[train_indexes],
                                         Y_train=df_ys[train_indexes],
                                         X_val=df_xs[valid_indexes],
                                         Y_val=df_ys[valid_indexes],
                                         export_model_path=_PATH_FOR_NN_MODEL + 'pollutant_tile_model_' + str(
                                             epochs) + '_epochs.h5',
                                         export_model_name_path=_PATH_FOR_NN_MODEL_NAME + 'pollutant_tile_model_name' + str(
                                             epochs) + '_epochs.h5',
                                         epochs=epochs)
    end_time = dt.datetime.now()
    print()
    print('Process finished!')
    print('Started At: ', start_time)
    print('Ended At: ', end_time)
    print('Run for ', end_time - start_time, ' time!')

# ---------------------------------------------- #
# ---------- 6b) Test Unet with Tiles ---------- #
# ---------------------------------------------- #

if _FLAG_FOR_TESTING_UNET:
    epochs = _EPOCHS
    nn_uNet = my_uNet.MyUNet()
    nn_uNet.set_uNet(height=_TILE_SIZE, width=_TILE_SIZE, channels_input=10, channels_output=2, n_filters=16)
    path_to_model = _PATH_FOR_NN_MODEL + 'pollutant_tile_model_' + str(epochs) + '_epochs.h5'
    nn_uNet.load_model(path_to_model)

    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    covid_measures_headers = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                              'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

    import_img_path_folder = _PATH_FOR_SAT_IMAGE_TILES
    import_img_path_covid_folder = _PATH_FOR_COVID_MEASURE_TILES
    import_test_indexes = xport_train_val_test_data_path_with_epochs = _PATH_FOR_TRAIN_VAL_TEST_INDEXES + 'epochs_' + str(
        epochs).zfill(4) + '/'

    list_country_dir = os.listdir(import_test_indexes)

    total_score = [0, 0]
    total_div_score = 0
    total_score_str_list = []
    # for in_country in list_country_dir:
    export_file_tmp = open('export_data/uNet_Test_Scores.txt', 'a')
    for in_country in ['Albania']:
        country = in_country.replace('_', ' ')
        print('\n\n\n')
        print('Test uNet for ' + country + ' ' + dt.datetime.now().strftime("(%Y-%m-%d %H:%M:%S)"))
        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=_LIST_DATE_RANGES_FOR_PATHS)

        dict_with_covid_paths, dict_covid_week_start_end_ranges, dict_covid_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_covid_folder + country.replace(' ', '_') + '/',
            list_pollution=covid_measures_headers,
            list_date_range=_LIST_DATE_RANGES_FOR_PATHS)

        df_xs = []
        df_ys = []
        # print(len(dict_with_dtile_paths[pollution_keywords_in_path[0]]))
        # print(len(dict_covid_export_index[covid_measures_headers[0]]))
        min_keywords_len = None
        for keyword in pollution_keywords_in_path:
            if min_keywords_len is None:
                min_keywords_len = len(dict_with_tile_paths[keyword])
            else:
                if len(dict_with_tile_paths[keyword]) < min_keywords_len:
                    min_keywords_len = len(dict_with_tile_paths[keyword])

        min_covid_keywords_len = None
        for keyword in covid_measures_headers:
            if min_covid_keywords_len is None:
                min_covid_keywords_len = len(dict_with_covid_paths[keyword])
            else:
                if len(dict_with_covid_paths[keyword]) < min_keywords_len:
                    min_covid_keywords_len = len(dict_with_covid_paths[keyword])

        if min_keywords_len != 0 and min_covid_keywords_len != 0:
            for paths_id in range(0, min_keywords_len):
                tmp_xs = []
                tmp_ys = []
                for keyword in pollution_keywords_in_path:
                    if keyword is pollution_keywords_in_path[0]:
                        for covid_keyword in covid_measures_headers:
                            for covid_id in range(min_covid_keywords_len):
                                if dict_week_start_end_ranges[keyword][paths_id] == \
                                        dict_covid_week_start_end_ranges[covid_keyword][covid_id]:
                                    # print(dict_week_start_end_ranges[keyword][paths_id], dict_covid_week_start_end_ranges[covid_keyword][covid_id])
                                    x_img = my_geo_img.open_image_file(
                                        dict_with_covid_paths[covid_keyword][covid_id][0])
                                    tmp_xs.append(x_img)
                                    break

                    x_img = my_geo_img.open_image_file(dict_with_tile_paths[keyword][paths_id][0])  # this week
                    y_img = my_geo_img.open_image_file(dict_with_tile_paths[keyword][paths_id][1])  # next week
                    tmp_xs.append(x_img)
                    tmp_ys.append(y_img)
                tmp_xs = np.array(tmp_xs).T
                tmp_ys = np.array(tmp_ys).T
                df_xs.append(tmp_xs)
                df_ys.append(tmp_ys)

            df_xs = np.array(df_xs) / 255.0
            df_ys = np.array(df_ys) / 255.0

            test_indexes = pd.read_csv(import_test_indexes + in_country + '/' + in_country + '_' +
                                       'test_indexes.csv')['test_indexes'].values.tolist()

            country_score = nn_uNet.test_uNet(df_xs[test_indexes], df_ys[test_indexes])
            total_score += country_score
            total_div_score += 1
            message = country + '_score= ' + ''.join(str(e) + ', ' for e in country_score)
            export_file_tmp.write(country + '_score= ')
