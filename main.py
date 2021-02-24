import pandas as pd  # import pandas for array processing
import matplotlib.pyplot as plt  # import pyplot for figure plotting
from my_lib import my_calendar_v2 as my_cal_v2  # import my calendar lib (for date anaysis)
import my_lib.my_geo_img as my_geo_img  # import my_geo_img lib (for image processing)
from google_lib import my_gee  # import my_gee lib (for satellite images)
import time  # import time
import os  # import os
import numpy as np
import test_uNet as my_uNet
import datetime as dt

pd.options.mode.chained_assignment = None  # this line prevents an error of pd.fillna() function


# ----------------------------------------- #
# ---------- 1) Find Date Ranges ---------- #
# ----------------------------------------- #


def check_create_folders_in_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


# path_data_fig_plot = 'export_data/img_data_plots/'  # the path to export the figures
# path_sat_stats_data_fig_plot = 'export_data/img_sat_stats_plot/'  # the path to export the figures
tile_size = 32
path_for_saving_plots = 'export_data/figure_plots/'
path_for_saving_tiles = 'D:/Documents/Didaktoriko/ACNN/export_data/figure_tiles_' + str(tile_size) + 'x' + str(
    tile_size) + '/'
check_create_folders_in_path(path_for_saving_tiles)
path_for_saving_dx_tiles = 'D:/Documents/Didaktoriko/ACNN/export_data/figure_dtiles_' + str(tile_size) + 'x' + str(
    tile_size) + '/'
check_create_folders_in_path(path_for_saving_dx_tiles)

export_model_covid_meas_path = 'D:/Documents/Didaktoriko/ACNN/export_data/covid_measures_' + str(tile_size) + 'x' + str(
    tile_size) + '/'
check_create_folders_in_path(export_model_covid_meas_path)

export_model_path = 'export_data/uNet/export_model/'
check_create_folders_in_path(export_model_path)
export_model_name_path = 'export_data/uNet/export_model_name/'
check_create_folders_in_path(export_model_name_path)
export_train_val_test_data_path = 'D:/Documents/Didaktoriko/ACNN/export_data/train_val_test_' + str(
    tile_size) + 'x' + str(
    tile_size) + '/'
check_create_folders_in_path(export_train_val_test_data_path)

flag_print_covid_country_plot_fig = False  # a flag for time saving purposes
flag_download_satellite_data = False  # a flag for time saving purposes
flag_break_images_to_tiles = False  # a flag for time saving purposes
flag_find_dx_between_images = False  # a flag for time saving purposes
flag_covid_meas_images = False  # a flag for time saving purposes
flag_train_UNet = False  # a flag for time saving purposes
flag_test_UNet = True  # a flag for time saving purposes

start_date = "2020-01-01"  # "YYYY-MM-DD"
end_date = "2020-12-31"  # "YYYY-MM-DD"

EPOCHS = 5  # the epochs for uNet (later usage)

# Create period ster a multiple of week period (by week I dont mean Monday to Sunday necessary, but a 7 day in a row)
period_alpha = 1  # a multiplier for weeks
period_step = period_alpha * 6  # the period step

# Break the date range into smaller ranges
list_date_period_ranges = my_cal_v2.break_date_range_to_periods(date_start=start_date, date_end=end_date,
                                                                period_step=period_step,
                                                                date_format=my_cal_v2.YYYY_MM_DD,
                                                                date_delimeter=my_cal_v2.del_dash, century=21)

# Create a list with stings of the date range.
# e.g. ['2020-01-01', '2020-01-07'] => '20200101_20200107'
list_date_range_path_string = my_cal_v2.create_string_list_date_range(list_input=list_date_period_ranges,
                                                                      del_input=my_cal_v2.del_dash,
                                                                      del_output=my_cal_v2.del_none)

# print(list_date_period_ranges)
# print(list_date_range_path_string)

# ------------------------------------------------- #
# ---------- 3) Create Datasets From CSV ---------- #
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


def clear_dir_files(path):
    import shutil
    if os.path.exists(path):
        list_dir = os.listdir(path)
        for dir_ in list_dir:
            print("Removing " + path + dir_)
            shutil.rmtree(path + dir_)


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


str_covid_measures_csv_path = "Data/covid_measures.csv"  # path to covid measure excel
df_covid_measures = pd.read_csv(str_covid_measures_csv_path, low_memory=False)  # read the csv file
# print(df_covid_measures.head())

list_unique_countries = df_covid_measures.COUNTRY.unique()  # create a list with the countries included in list
list_unique_countries_for_path = []
for country in list_unique_countries:
    list_unique_countries_for_path.append(country.replace(" ", "_"))
# print(list_unique_countries)
# print(len(list_unique_countries))

# df_pollution_dataset = df_covid_measures[['COUNTRY', 'ID', 'DATE', 'RESTRICTIONS_INTERNAL_MOVEMENTS',
#                                           'INTERNATIONAL_TRAVEL_CONTROLS', 'CONTINENT', 'POPULATION',
#                                           'POPULATION_DENSITY', 'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS',
#                                           'CLOSE_PUBLIC_TRANSPORT', 'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS',
#                                           'WORKPLACE_CLOSURES']]

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

# Create a list of headers for (used later for the final array)
df_pollution_data_header = ['DATE_RANGE', 'RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                            'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                            'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

# df_x_plot and df_y_plot used to export plots (for data visualization)
df_x_plot = 'DATE_RANGE'
df_y_plot = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
             'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
             'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

# fillna
df_pollution_dataset.fillna(method='ffill', inplace=True)
# create a dictionary from the list (using the countries as a key column)
df_polllution_dict = create_dictionary_from_list_column(list_input=df_pollution_dataset.values.tolist(),
                                                        key_column_index=0)

# print(df_polllution_dict.keys())
df_pollution_mean_range_dict = {}
df_pollution_mean_range_dict_with_headers = {}
if flag_print_covid_country_plot_fig:
    for key in df_polllution_dict.keys():
        df_pollution_mean_range_dict[key.replace(' ', '_')] = []
        df_pollution_mean_range_dict[key.replace(' ', '_')] = my_cal_v2.merge_values_in_date_range_list(
            list_input=df_polllution_dict.copy()[key],
            date_index=0,
            date_range_list=list_date_period_ranges,
            merge_type=my_cal_v2.merge_mean,
            del_input=my_cal_v2.del_dash,
            del_output=my_cal_v2.del_none,
            del_use=True)

        # print(key, df_pollution_mean_range_dict[key])

    for key in df_pollution_mean_range_dict.keys():
        path_to_export_plot_png = path_for_saving_plots + key + '/'
        if not os.path.exists(path_to_export_plot_png):
            os.mkdir(path_to_export_plot_png)
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')] = []
        for dataset_index in range(0, len(df_pollution_mean_range_dict[key])):
            tmp_dataset_list = [df_pollution_mean_range_dict[key][dataset_index][0]]
            for value in df_pollution_mean_range_dict[key][dataset_index][1]:
                tmp_dataset_list.append(value)
            df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')].append(tmp_dataset_list)
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')] = pd.DataFrame(
            df_pollution_mean_range_dict_with_headers[key],
            columns=df_pollution_data_header)
        df_pollution_mean_range_dict_with_headers[key.replace(' ', '_')].plot(x=df_x_plot, y=df_y_plot)
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

# --------------------------------------------------- #
# ---------- 3) Download Sentiner-5 Images ---------- #
# --------------------------------------------------- #

# my_gee.clear_tasks()
if flag_download_satellite_data:
    list_collection_id = ['COPERNICUS/S5P/OFFL/L3_O3', 'COPERNICUS/S5P/OFFL/L3_CO', 'COPERNICUS/S5P/OFFL/L3_AER_AI',
                          'COPERNICUS/S5P/OFFL/L3_HCHO', 'COPERNICUS/S5P/OFFL/L3_SO2',
                          'COPERNICUS/S5P/OFFL/L3_NO2']
    list_colection_bands = ['O3_column_number_density', 'CO_column_number_density', 'absorbing_aerosol_index',
                            'tropospheric_HCHO_column_number_density', 'SO2_column_number_density',
                            'NO2_column_number_density']
    list_collection_names = ['ozone_O3_density', 'carbon_monoxide_CO_density', 'absorbing_aerosol_index',
                             'offline_formaldehyde_HCHO_density', 'sulphur_dioxide_SO2_density',
                             'nitrogen_dioxide_NO2_density']

    list_S5_data_len = len(list_collection_id)
    scale_m_per_px = 5000
    waiting_minute_multiplier = 0
    waiting_time_in_sec = int(waiting_minute_multiplier) * 60
    # for i in range(0, list_S5_data_len):
    #     my_gee.download_image_from_collection(collection_id=list_collection_id[i],
    #                                           image_band=list_colection_bands[i],
    #                                           img_name=list_collection_names[i],
    #                                           list_date_range=list_date_period_ranges,
    #                                           list_countries=list_unique_countries,
    #                                           scale=scale_m_per_px,
    #                                           waiting_time=waiting_time_in_sec)
    my_gee.download_image_from_collection(collection_id=list_collection_id[5],
                                          image_band=list_colection_bands[5],
                                          img_name=list_collection_names[5],
                                          list_date_range=list_date_period_ranges,
                                          list_countries=list_unique_countries,
                                          scale=scale_m_per_px,
                                          waiting_time=waiting_time_in_sec)

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

if flag_break_images_to_tiles:
    clear_dir_files(path_for_saving_tiles)
    print()
    img_path_folder = "Data/Satellite_Atmospheric_Images/tiff_folders/GEE_"
    # pollution_keywords_in_path = ['carbon_monoxide', 'ozone', "sulphur_dioxide", "nitrogen_dioxide"]
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    pollution_dict_min_max_values = {'carbon_monoxide': {'min': 0.00, 'max': 0.10},  # mol/m^2
                                     'ozone': {'min': 0.10, 'max': 0.18}}  # mol/m^2
    country_index = 1
    for country in list_unique_countries_for_path:
        print('(' + str(country_index) + ' / ' + str(len(list_unique_countries_for_path))
              + ') Exporting tiles for ' + country)
        country_index += 1
        tmp_img_path_folder = img_path_folder + country + "/"
        img_path_files = os.listdir(tmp_img_path_folder)
        path_to_export_tile_png = path_for_saving_tiles + country + '/'
        if not os.path.exists(path_to_export_tile_png):
            os.mkdir(path_to_export_tile_png)
        for keyword in pollution_keywords_in_path:
            for date_range in list_date_range_path_string:
                for path_file in img_path_files:
                    if keyword in path_file and date_range in path_file:
                        # print(tmp_img_path_folder + path_file)
                        img = my_geo_img.open_geospatial_image_file(tmp_img_path_folder + path_file)
                        my_geo_img.img_break_tiles_and_export_them(img=img,
                                                                   tile_size=tile_size,
                                                                   folder_path=path_to_export_tile_png,
                                                                   export_name=country + '_' + keyword + '_' + date_range,
                                                                   norm_min_value=
                                                                   pollution_dict_min_max_values[keyword]['min'],
                                                                   normalize_max_value=
                                                                   pollution_dict_min_max_values[keyword]['max'],
                                                                   info_acc_percent=0.6)

# --------------------------------------------------------- #
# ---------- 5b) Break Satelite Images to DTiles ---------- #
# --------------------------------------------------------- #

if flag_find_dx_between_images:
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    import_img_path_folder = path_for_saving_tiles
    clear_dir_files(path_for_saving_dx_tiles)
    print()
    for country in list_unique_countries:
        # print()
        print('Exporting Dtiles for ' + country)
        export_dir_path = path_for_saving_dx_tiles + country.replace(' ', '_') + '/'
        if not os.path.exists(export_dir_path):
            os.makedirs(export_dir_path)

        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=list_date_range_path_string)
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
                        + d_date_range[1] + '_size_' + str(tile_size) + 'x' + str(tile_size) +
                        '_tile_' + str(export_index) + '_id' + '.png')

                my_geo_img.export_image_file(export_img_path, dtile_img)

# ----------------------------------------------- #
# ---------- 5c) Train Unet with Tiles ---------- #
# ----------------------------------------------- #
if flag_covid_meas_images:
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

    clear_dir_files(export_model_covid_meas_path)
    print()
    for country in list_unique_countries:
        print('Export covid measure images for ' + country)
        export_id = 0
        check_create_folders_in_path(export_model_covid_meas_path + country.replace(' ', '_') + '/')
        for date_range in list_date_range_path_string:
            for measure in covid_measures_headers:
                for i in range(0, len(df_acnn)):
                    if country.replace(' ', '_') == df_acnn[i][0] and date_range == df_acnn[i][1]:
                        norm_value = (df_dict_acnn[measure][i] / 4.0) * 255
                        tmp_measure_img = create_measure_img(norm_value, tile_size, tile_size)
                        img_export_path = (
                                export_model_covid_meas_path + country.replace(' ', '_') + '/' +
                                country.replace(' ', '_') + '_' + measure + '_' + date_range + '_' + 'size_' +
                                str(tile_size) + 'x' + str(tile_size) + '_tile_' + str(export_id) + '_id' + '.png')
                        my_geo_img.export_image_file(img_export_path, np.array(tmp_measure_img))
                        break

# ----------------------------------------------- #
# ---------- 6a) Train Unet with Tiles ---------- #
# ----------------------------------------------- #
if flag_train_UNet:
    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    covid_measures_headers = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                              'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

    epochs = EPOCHS
    nn_uNet = my_uNet.MyUNet()
    nn_uNet.set_uNet(height=tile_size, width=tile_size, channels_input=10, channels_output=2, n_filters=16)

    import_img_path_folder = path_for_saving_tiles
    import_img_path_covid_folder = export_model_covid_meas_path
    # country = 'Albania'
    tmp_results_index = 0
    export_train_val_test_data_path_with_epochs = export_train_val_test_data_path + 'epochs_' + str(epochs).zfill(
        4) + '/'
    check_create_folders_in_path(export_train_val_test_data_path_with_epochs)
    clear_dir_files(export_train_val_test_data_path_with_epochs)
    start_time = dt.datetime.now()
    for country in list_unique_countries:
        print('\n\n\n')
        print('Train uNet for ' + country + ' ' + dt.datetime.now().strftime("(%Y-%m-%d %H:%M:%S)"))
        print('Process started at: ', start_time.strftime("(%Y-%m-%d %H:%M:%S)"))
        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=list_date_range_path_string)

        dict_with_covid_paths, dict_covid_week_start_end_ranges, dict_covid_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_covid_folder + country.replace(' ', '_') + '/',
            list_pollution=covid_measures_headers,
            list_date_range=list_date_range_path_string)

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
                                         export_model_path=export_model_path + 'pollutant_tile_model_' + str(
                                             epochs) + '_epochs.h5',
                                         export_model_name_path=export_model_name_path + 'pollutant_tile_model_name' + str(
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

if flag_test_UNet:
    epochs = EPOCHS
    nn_uNet = my_uNet.MyUNet()
    nn_uNet.set_uNet(height=tile_size, width=tile_size, channels_input=10, channels_output=2, n_filters=16)
    path_to_model = export_model_path + 'pollutant_tile_model_' + str(epochs) + '_epochs.h5'
    nn_uNet.load_model(path_to_model)

    pollution_keywords_in_path = ['carbon_monoxide', 'ozone']
    covid_measures_headers = ['RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                              'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                              'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']

    import_img_path_folder = path_for_saving_tiles
    import_img_path_covid_folder = export_model_covid_meas_path
    import_test_indexes = xport_train_val_test_data_path_with_epochs = export_train_val_test_data_path + 'epochs_' + str(
        epochs).zfill(4) + '/'

    list_country_dir = os.listdir(import_test_indexes)

    total_score = [0, 0]
    total_div_score = 0
    total_score_str_list = []
    # for in_country in list_country_dir:
    import pickle
    export_file_tmp = open('export_data/uNet_Test_Scores.txt', 'a')
    for in_country in ['Albania']:
        country = in_country.replace('_', ' ')
        print('\n\n\n')
        print('Test uNet for ' + country + ' ' + dt.datetime.now().strftime("(%Y-%m-%d %H:%M:%S)"))
        dict_with_tile_paths, dict_week_start_end_ranges, dict_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_folder + country.replace(' ', '_') + '/',
            list_pollution=pollution_keywords_in_path,
            list_date_range=list_date_range_path_string)

        dict_with_covid_paths, dict_covid_week_start_end_ranges, dict_covid_export_index = my_geo_img.find_tile_paths_matches(
            list_img_path_dir=import_img_path_covid_folder + country.replace(' ', '_') + '/',
            list_pollution=covid_measures_headers,
            list_date_range=list_date_range_path_string)

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

