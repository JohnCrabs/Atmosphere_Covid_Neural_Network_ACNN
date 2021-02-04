import pandas as pd
# import matplotlib.pyplot as plt
from my_lib import my_calendar_v2 as my_cal_v2
from google_lib import my_gee

# ----------------------------------------- #
# ---------- 1) Find Date Ranges ---------- #
# ----------------------------------------- #

download_satellite_data = False

start_date = "2020-01-01"  # "YYYY-MM-DD"
end_date = "2020-12-31"  # "YYYY-MM-DD"

# Create period ster a multiple of week period (by week I dont mean Monday to Sunday necessary, but a 7 day in a row)
period_alpha = 1
period_step = period_alpha * 7

list_date_period_ranges = my_cal_v2.break_date_range_to_periods(date_start=start_date, date_end=end_date,
                                                                period_step=period_step,
                                                                date_format=my_cal_v2.YYYY_MM_DD,
                                                                date_delimeter=my_cal_v2.del_dash, century=21)
# print(list_date_period_ranges)


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
    list_output = []
    for item in list_input:
        if item not in list_output:
            list_output.append(item)
    return list_output


def create_dictionary_from_list_column(list_input: [], key_column_index: int):
    dict_output = {}
    for row in list_input:
        if row[key_column_index] not in dict_output.keys():
            dict_output[row[key_column_index]] = []
        tmp_list = []
        for index in range(0, len(row)):
            if index != key_column_index:
                tmp_list.append(row[index])
        dict_output[row[key_column_index]].append(tmp_list)
    return dict_output


str_covid_measures_csv_path = "Data/covid_measures.csv"
df_covid_measures = pd.read_csv(str_covid_measures_csv_path, low_memory=False)
# print(df_covid_measures.head())

list_unique_countries = df_covid_measures.COUNTRY.unique()
# print(list_unique_countries)
# print(len(list_unique_countries))

# df_pollution_dataset = df_covid_measures[['COUNTRY', 'ID', 'DATE', 'RESTRICTIONS_INTERNAL_MOVEMENTS',
#                                           'INTERNATIONAL_TRAVEL_CONTROLS', 'CONTINENT', 'POPULATION',
#                                           'POPULATION_DENSITY', 'CANCEL_PUBLIC_EVENTS', 'RESTRICTION_GATHERINGS',
#                                           'CLOSE_PUBLIC_TRANSPORT', 'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS',
#                                           'WORKPLACE_CLOSURES']]

df_pollution_dataset = df_covid_measures[['COUNTRY', 'DATE', 'RESTRICTIONS_INTERNAL_MOVEMENTS',
                                          'INTERNATIONAL_TRAVEL_CONTROLS', 'CANCEL_PUBLIC_EVENTS',
                                          'RESTRICTION_GATHERINGS', 'CLOSE_PUBLIC_TRANSPORT',
                                          'SCHOOL_CLOSURES', 'STAY_HOME_REQUIREMENTS', 'WORKPLACE_CLOSURES']]

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

df_pollution_dataset.fillna(method='ffill', inplace=True)
df_polllution_dict = create_dictionary_from_list_column(list_input=df_pollution_dataset.values.tolist(),
                                                        key_column_index=0)

# print(df_polllution_dict.keys())

df_pollution_mean_range_dict = {}
for key in df_polllution_dict.keys():
    df_pollution_mean_range_dict[key] = []
    df_pollution_mean_range_dict[key] = my_cal_v2.merge_values_in_date_range_list(
        list_input=df_polllution_dict.copy()[key],
        date_index=0,
        date_range_list=list_date_period_ranges,
        merge_type=my_cal_v2.merge_mean)

    # print(key, df_pollution_mean_range_dict[key])

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
                             'offline_formaldehyde_HCHO_density', 'sulphur_dioxide_SO2_density'
                                                                  'nitrogen_dioxide_NO2_density']

    list_S5_data_len = len(list_collection_id)
    scale_m_per_px = 5000
    waiting_minute_multiplier = 5
    waiting_time_in_sec = int(waiting_minute_multiplier) * 60
    for i in range(0, list_S5_data_len):
        my_gee.download_image_from_collection(collection_id=list_collection_id[i],
                                              image_band=list_colection_bands[i],
                                              img_name=list_collection_names[i],
                                              list_date_range=list_date_period_ranges,
                                              list_countries=list_unique_countries,
                                              scale=scale_m_per_px,
                                              waiting_time=waiting_time_in_sec)
