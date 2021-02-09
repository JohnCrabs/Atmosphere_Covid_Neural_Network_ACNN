import pandas as pd  # import pandas for array processing
import matplotlib.pyplot as plt  # import pyplot for figure plotting
from my_lib import my_calendar_v2 as my_cal_v2  # import my calendar lib (for date anaysis)
import my_lib.my_geo_img as my_geo_img  # import my_geo_img lib (for image processing)
from google_lib import my_gee  # import my_gee lib (for satellite images)
import time  # import time
import os  # import os

pd.options.mode.chained_assignment = None  # this line prevents an error of pd.fillna() function

# ----------------------------------------- #
# ---------- 1) Find Date Ranges ---------- #
# ----------------------------------------- #

# path_data_fig_plot = 'export_data/img_data_plots/'  # the path to export the figures
# path_sat_stats_data_fig_plot = 'export_data/img_sat_stats_plot/'  # the path to export the figures
path_for_saving_plots = 'export_data/figure_plots/'

print_covid_country_plot_fig = True  # a flag for time saving purposes
download_satellite_data = False  # a flag for time saving purposes

start_date = "2020-01-01"  # "YYYY-MM-DD"
end_date = "2020-12-31"  # "YYYY-MM-DD"

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
if print_covid_country_plot_fig:
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
if download_satellite_data:
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

# ----------------------------------------------- #
# ---------- 4) Read Sentiner-5 Images ---------- #
# ----------------------------------------------- #

df_satelite_image_data_dict = {}
tmp_test_country = ['Greece']
pollution_statistic_header = ['DATE_RANGE', 'MIN', 'MAX', 'MEAN', 'STD_DEV', 'MEDIAN']
pollution_keywords_in_path = ['carbon_monoxide', 'ozone', "sulphur_dioxide", "nitrogen_dioxide"]
if print_covid_country_plot_fig:
    img_path_folder = "Data/Satellite_Atmospheric_Images/tiff_folders/GEE_"

    for country in list_unique_countries_for_path:
        # for country in tmp_test_country:
        tmp_img_path_folder = img_path_folder + country + "/"
        img_path_files = os.listdir(tmp_img_path_folder)

        path_to_export_plot_png = path_for_saving_plots + country + '/'
        if not os.path.exists(path_to_export_plot_png):
            os.mkdir(path_to_export_plot_png)

        dict_pollution_stats = {}
        for keyword in pollution_keywords_in_path:
            dict_pollution_stats[keyword] = []
            for date_range in list_date_range_path_string:
                for path_file in img_path_files:
                    if keyword in path_file and date_range in path_file:
                        img = my_geo_img.open_image(tmp_img_path_folder + path_file)
                        img_stats = my_geo_img.img_statistics(img, round_at=5)

                        tmp_stats_list = [date_range,
                                          img_stats[my_geo_img.MY_GEO_MIN],
                                          img_stats[my_geo_img.MY_GEO_MAX],
                                          img_stats[my_geo_img.MY_GEO_MEAN],
                                          img_stats[my_geo_img.MY_GEO_STD_DEV],
                                          img_stats[my_geo_img.MY_GEO_MEDIAN]]

                        dict_pollution_stats[keyword].append(tmp_stats_list)
            df_pollution_stats = pd.DataFrame(dict_pollution_stats[keyword], columns=pollution_statistic_header)
            df_pollution_stats.plot(x=df_x_plot, y=['MIN', 'MAX', 'MEAN', 'STD_DEV', 'MEDIAN'])
            plt.gcf().set_size_inches(20.48, 10.24)
            plt.xticks(rotation=90)
            plt.xlim(0, 53)
            # plt.ylim(0, 6)
            plt.title(country + "_" + keyword + "_" + "density")
            plt.xlabel('Date_Range')
            plt.ylabel('Density')
            plt.xticks(range(0, 53))
            plt.savefig(path_to_export_plot_png + country + "_" + keyword + ".png", dpi=100)
            time.sleep(1)
            print("Plot " + country + " " + keyword + " exported")
            df_satelite_image_data_dict[country] = dict_pollution_stats.copy()

# print(dict_pollution_stats)

# img = geo_img.open_image(img_path)
# img_stats = geo_img.img_statistics(img, round_at=5)
# print(img_stats)

# --------------------------------------------- #
# ---------- 5) Create Exporting CSV ---------- #
# --------------------------------------------- #

# print(df_pollution_mean_range_dict)
# print(df_satelite_image_data_dict)
csv_output_list = [['COUNTRY', 'DATE_RANGE', 'RESTRICTIONS_INTERNAL_MOVEMENTS', 'INTERNATIONAL_TRAVEL_CONTROLS',
                    'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT', 'SCHOOL_CLOSURES',
                    'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES', 'CARBON_MONOOXIDE_MAX', 'CARBON_MONOOXIDE_MIN',
                    'CARBON_MONOOXIDE_MEAN', 'CARBON_MONOOXIDE_STDDEV', 'CARBON_MONOOXIDE_MEDIAN', 'OZONE_MAX',
                    'OZONE_MIN', 'OZONE_MEAN', 'OZONE_STDDEV', 'OZONE_MEDIAN', 'SULPHUR_DIOXIDE_MAX',
                    'SULPHUR_DIOXIDE_MIN', 'SULPHUR_DIOXIDE_MEAN', 'SULPHUR_DIOXIDE_STDDEV', 'SULPHUR_DIOXIDE_MEDIAN',
                    'NITROGEN_DIOXIDE_MAX', 'NITROGEN_DIOXIDE_MIN', 'NITROGEN_DIOXIDE_MEAN', 'NITROGEN_DIOXIDE_STDDEV',
                    'NITROGEN_DIOXIDE_MEDIAN']]

for country in list_unique_countries_for_path:
    # for country in tmp_test_country:
    for date_range in list_date_range_path_string:
        tmp_append_list = [country, date_range]
        for row in df_pollution_mean_range_dict[country]:
            if row[0] == date_range:
                for value in row[1]:
                    tmp_append_list.append(value)
        for key in pollution_keywords_in_path:
            for row in df_satelite_image_data_dict[country][key]:
                if date_range in row:
                    for index in range(1, len(row)):
                        tmp_append_list.append(row[index])
        csv_output_list.append(tmp_append_list)

csv_path = "export_data/acnn_data.csv"
my_cal_v2.write_csv(csv_path=csv_path, list_write=csv_output_list, delimeter=my_cal_v2.del_comma)
