"""
my_calendar_v2.py contains the Class My_Calendar, which creates a dictionary with events.

Calendar Structures:
1) Dict{Date_Format: {Time_Format: {Event: some_type_of_value}}}
2) Dict{Date_Format: {Event: some_type_of_value}}

"some_type_of_value" can be: int, float, double, str, list, dict, etc

Example 1: Let's say we need to add a temperature event, we can do it like this.
Dict{'08/01/2020': {'00:00': {'Athens': {Date: '08/01/2020', Time: '00:00', Temperature: '26.2',
                              'Brest': {Date: '08/01/2020', Time: '00:00', Temperature: '16.6'}}}}

Let's say we need to export it to excel:
1) Give date range for the exporting dates (e.g. start: 08/01/2020, end: 11/30/2020)
2) Give the format of the file (CSV of XLSX)
3) Create First Line ['Date', 'Time', 'Athens_Temperature', 'Brest_Temperature]
4) For each date in range export these value. If value does not exist print -9999.99
"""

import datetime as dt
import warnings

# Create lists that corresponds to date, time strings
# Create list for months
list_str_id_months_m = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
list_str_id_months_mm = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

# Create list for days
list_str_id_days_d = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
                      '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                      '25', '26', '27', '28', '29', '30', '31']
list_str_id_days_dd = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
                       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
                       '25', '26', '27', '28', '29', '30', '31']

# Create list for hours
list_str_id_hours = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11',
                     '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']

# Create list for hours
list_str_id_minutes = ['00', '01', '02', '03', '04', '05', '06', '07', '08', '09',
                       '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
                       '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                       '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                       '40', '41', '42', '43', '44', '45', '46', '47', '48', '49',
                       '50', '51', '52', '53', '54', '55', '56', '57', '58', '59']

# Leap situations
list_int_month_days_not_leap = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
list_int_month_days_leap = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Different systems to create a date
DD_MM_YYYY = 'ddmmyyyy'
DD_YYYY_MM = 'ddyyyymm'
MM_DD_YYYY = 'mmddyyyy'
MM_YYYY_DD = 'mmyyyydd'
YYYY_MM_DD = 'yyyymmdd'
YYYY_DD_MM = 'yyyyddmm'

D_M_YYYY = 'dmyyyy'
D_YYYY_M = 'dyyyym'
M_D_YYYY = 'mdyyyy'
M_YYYY_D = 'myyyyd'
YYYY_M_D = 'yyyymd'
YYYY_D_M = 'yyyydm'

DD_MM_YY = 'ddmmyy'
DD_YY_MM = 'ddyymm'
MM_DD_YY = 'mmddyy'
MM_YY_DD = 'mmyyydd'
YY_MM_DD = 'yymmdd'
YY_DD_MM = 'yyddmm'

D_M_YY = 'dmyy'
D_YY_M = 'dyym'
M_D_YY = 'mdyy'
M_YY_D = 'myyd'
YY_M_D = 'yymd'
YY_D_M = 'yydm'

# Date Formats
SINGLE = 'single'
DOUBLE = 'double'

# Different systems to create the hour
HH_MM = 'hhmm'
HH_MM_SS = 'hhmmss'

# Delimeters
del_none = ''
del_comma = ','
del_hashtag = '#'
del_colon = ':'
del_semicolon = ';'
del_space = ' '
del_underscore = '_'
del_dash = '-'
del_slash = '/'

NaN = "nan"
EMPTY_LIST = []
EMPTY_DICT = {}

merge_min = 'min'
merge_max = 'max'
merge_mean = 'mean'
merge_median = 'median'

# ------------------------------- #
# ---------- FUNCTIONS ---------- #
# ------------------------------- #
# -------------------------------- #
# ---------- 1) I/O CSV ---------- #
# -------------------------------- #


def read_csv(csv_path: str, delimeter=del_comma):
    """
    A Function for reading CSV files.
    :param csv_path: The path for the csv to be imported.
    :param delimeter: The delimeter of values.
    :return: []
    """
    import csv  # import csv library
    list_tmp_csv = []  # create a temporary empty list
    with open(csv_path) as csv_file:  # open the csv path
        csv_reader = csv.reader(csv_file, delimiter=delimeter)
        for row in csv_reader:  # for each row in csv_reader file
            list_tmp_csv.append(row)  # append the row to list
    return list_tmp_csv  # return the list


def write_csv(csv_path: str, list_write: [], delimeter=del_comma):
    """
    A Function for writing CSV files.
    :param csv_path: The path for the csv to be exported.
    :param list_write: The list to be written in file
    :param delimeter: The delimeter for values
    :return: Nothing
    """
    import csv  # import csv library
    import os  # import os library
    if not os.path.exists(os.path.dirname(csv_path)):  # Check if path does not exist
        warnings.warn("Path does not exist! Path will be created!")  # Warn in console
        directory = os.path.dirname(csv_path)  # Find the direcrories in the path
        os.mkdir(directory)  # Create the directories

    with open(csv_path, 'w', newline='') as csvfile:  # Open the path to write the file
        csv_writer = csv.writer(csvfile, delimiter=delimeter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in list_write:  # For each row
            csv_writer.writerow(row)  # Write row to file


# ------------------------------------------- #
# ---------- 2) Calendar Callables ---------- #
# ------------------------------------------- #

def set_time(hour_start=0, hour_end=24, hour_step=1, minute_start=0, minute_end=60, minute_step=0,
             delimeter=del_colon):
    """
    :param hour_start: The starting hour. Range (0, 24)
    :param hour_end: The ending hour. Range (0, 24)
    :param hour_step: The step for hours. Range (1, 24). Big step means same hour.
    :param minute_start: The starting minute. Range (0, 60).
    :param minute_end: The ending minute. Range (0, 60)
    :param minute_step: The minute step. Range (0, 60). Big step means the same minute every hour.
    :param delimeter: Delimeter for time
    :return: list_timestamp, int_size
    """
    if hour_start < 0 or hour_start > 24:  # if hour_start outside original hour ranges
        warnings.warn("Hour start out of range. Set it to default value '0'")
        hour_start = 0  # set hour_start to default value
    if hour_end < hour_start or hour_end > 24:  # if hour_end outside original hour ranges
        warnings.warn("Hour end out of range. Set it to default value '24'")
        hour_end = 24  # set hour_end to default value
    if minute_start < 0 or minute_start > 60:  # if minute_start outside original minutes ranges
        warnings.warn("Minute start out of range. Set it to default value '0'")
        minute_start = 0  # set minute_start to default value
    if minute_end < minute_start or minute_end > 60:  # if minute_end outside original minutes ranges
        warnings.warn("Minute end out of range. Set it to default value '60'")
        minute_end = 60  # set minute_end to default value

    list_timestamp = []  # create list_timestamp
    int_size = 0  # create an integer to the size of the list_timestamp

    if hour_step <= 0:  # if hour_step less equals to 0 return [], 0
        return list_timestamp, int_size
    else:
        for hour in range(hour_start, hour_end, hour_step):
            if minute_step == 0:  # if minute_step is 0
                # Create a timestamp_tmp
                timestamp_tmp = list_str_id_hours[hour] + delimeter
                timestamp_tmp += list_str_id_minutes[0]
                list_timestamp.append(timestamp_tmp)  # append to list_timestamp
                int_size += 1
                # print(timestamp_tmp)
            else:
                for minutes in range(minute_start, minute_end, minute_step):
                    # Create a timestamp_tmp
                    timestamp_tmp = list_str_id_hours[hour] + delimeter
                    timestamp_tmp += list_str_id_minutes[minutes]
                    list_timestamp.append(timestamp_tmp)  # append to list_timestamp
                    int_size += 1
                    # print(timestamp_tmp)
    # print(list_timestamp)
    return list_timestamp, int_size


def change_day_month_format(day, month, date_format=DOUBLE):
    """
    For example 01/01/2021 to 1/1/2021
    :param day: a string of day
    :param month: a string of month
    :param date_format: the format needs to transform
    :return: day_format, month_format
    """
    day_format = int(day)  # cast the str to int for day
    month_format = int(month)  # cast the str to int for month
    if date_format is SINGLE:
        day_format = list_str_id_days_d[day_format - 1]  # recasting the int to str for the correct format for day
        month_format = list_str_id_months_m[month_format - 1]  # cast the str to int for month
    else:
        day_format = list_str_id_days_dd[day_format - 1]  # recasting the int to str for the correct format for day
        month_format = list_str_id_months_mm[month_format - 1]  # cast the str to int for month
    return day_format, month_format


def date_break_to_day_month_year(date: str, date_format: str, date_delimeter=del_slash, century=21):
    """
            Check the format of the day (eg DD/MM/YYYY) and returns a string.
            :param century: The current century
            :param date: The date to be changed
            :param date_format: The date format. Can be: DD_MM_YYYY
                                                         DD_YYYY_MM
                                                         MM_DD_YYYY
                                                         MM_YYYY_DD
                                                         YYYY_MM_DD
                                                         YYYY_DD_MM
                                                         D_M_YYYY
                                                         D_YYYY_M
                                                         M_D_YYYY
                                                         M_YYYY_D
                                                         YYYY_M_D
                                                         YYYY_D_M
                                                         DD_MM_YY
                                                         DD_YY_MM
                                                         MM_DD_YY
                                                         MM_YY_DD
                                                         YY_MM_DD
                                                         YY_DD_MM
                                                         D_M_YY
                                                         D_YY_M
                                                         M_D_YY
                                                         M_YY_D
                                                         YY_M_D
                                                         YY_D_M

            :param date_delimeter: The delimeter of the date.
            :return: day: str, month: str, year: str
            """
    # DD_MM_YYYY or D_M_YYYY or DD_MM_YY or D_M_YY
    if date_format == DD_MM_YYYY or date_format == D_M_YYYY or date_format == DD_MM_YY or date_format == D_M_YY:
        day, month, year = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == DD_MM_YY or date_format == D_M_YY:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year
    # DD_YYYY_MM or D_YYYY_M or DD_YY_MM or D_YY_M
    elif date_format == DD_YYYY_MM or date_format == D_YYYY_M or date_format == DD_YY_MM or date_format == D_YY_M:
        day, year, month = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == DD_YY_MM or date_format == D_YY_M:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year
    # MM_DD_YYYY or M_D_YYYY or MM_DD_YY or M_D_YY
    elif date_format == MM_DD_YYYY or date_format == M_D_YYYY or date_format == MM_DD_YY or date_format == M_D_YY:
        month, day, year = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == MM_DD_YY or date_format == M_D_YY:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year
    # MM_YYYY_DD or M_YYYY_D or MM_YY_DD or M_YY_D
    elif date_format == MM_YYYY_DD or date_format == M_YYYY_D or date_format == MM_YY_DD or date_format == M_YY_D:
        month, year, day = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == MM_YY_DD or date_format == M_YY_D:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year
    # YYYY_MM_DD or YYYY_M_D or YY_MM_DD or YY_M_D
    elif date_format == YYYY_MM_DD or date_format == YYYY_M_D or date_format == YY_MM_DD or date_format == YY_M_D:
        year, month, day = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == YY_MM_DD or date_format == YY_M_D:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year
    # YYYY_DD_MM or YYYY_D_M or YY_DD_MM or YY_D_M
    elif date_format == YYYY_DD_MM or date_format == YYYY_D_M or date_format == YY_DD_MM or date_format == YY_D_M:
        year, day, month = date.split(date_delimeter)  # split the date using the delimeter
        if date_format == YY_DD_MM or date_format == YY_D_M:
            year = str(int(year) + (century - 1) * 100)
        return day, month, year


def change_hour_number_list_to_time(list_hour: [], time_format=HH_MM):
    list_hour_tmp = []  # create tmp hour list
    if time_format == HH_MM:
        for hour in list_hour:
            list_hour_tmp.append('%02i' % int(hour) + ':00')
    elif time_format == HH_MM_SS:
        for hour in list_hour:
            list_hour_tmp.append('%02i' % int(hour) + ':00:00')
    return list_hour_tmp


def set_date_format(day: str, month: str, year: str, date_format=DD_MM_YYYY, date_delimeter=del_slash):
    """
    Check the format of the day (eg DD/MM/YYYY) and returns a string.
    :param day: The day of the date.
    :param month: The month of the date.
    :param year: The year of the date.
    :param date_format: The date format. Can be: DD_MM_YYYY
                                                 DD_YYYY_MM
                                                 MM_DD_YYYY
                                                 MM_YYYY_DD
                                                 YYYY_MM_DD
                                                 YYYY_DD_MM
                                                 D_M_YYYY
                                                 D_YYYY_M
                                                 M_D_YYYY
                                                 M_YYYY_D
                                                 YYYY_M_D
                                                 YYYY_D_M
                                                 DD_MM_YY
                                                 DD_YY_MM
                                                 MM_DD_YY
                                                 MM_YY_DD
                                                 YY_MM_DD
                                                 YY_DD_MM
                                                 D_M_YY
                                                 D_YY_M
                                                 M_D_YY
                                                 M_YY_D
                                                 YY_M_D
                                                 YY_D_M

    :param date_delimeter: The delimeter of the date.
    :return: A date string.
    """
    # DD_MM_YYYY
    if date_format == DD_MM_YYYY:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return day + date_delimeter + month + date_delimeter + year
    # DD_YYYY_MM
    elif date_format == DD_YYYY_MM:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return day + date_delimeter + year + date_delimeter + month
    # MM_DD_YYYY
    elif date_format == MM_DD_YYYY:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return month + date_delimeter + day + date_delimeter + year
    # MM_YYYY_DD
    elif date_format == MM_YYYY_DD:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return month + date_delimeter + year + date_delimeter + day
    # YYYY_MM_DD
    elif date_format == YYYY_MM_DD:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return year + date_delimeter + month + date_delimeter + day
    # YYYY_DD_MM
    elif date_format == YYYY_DD_MM:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return year + date_delimeter + day + date_delimeter + month

    # D_M_YYYY
    elif date_format == D_M_YYYY:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return day + date_delimeter + month + date_delimeter + year
    # D_YYYY_M
    elif date_format == D_YYYY_M:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return day + date_delimeter + year + date_delimeter + month
    # M_D_YYYY
    elif date_format == M_D_YYYY:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return month + date_delimeter + day + date_delimeter + year
    # M_YYYY_D
    elif date_format == M_YYYY_D:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return month + date_delimeter + year + date_delimeter + day
    # YYYY_M_D
    elif date_format == YYYY_M_D:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return year + date_delimeter + month + date_delimeter + day
    # YYYY_D_M
    elif date_format == YYYY_D_M:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return year + date_delimeter + day + date_delimeter + month

    # DD_MM_YY
    if date_format == DD_MM_YY:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return day + date_delimeter + month + date_delimeter + str(int(year) % 100)
    # DD_YY_MM
    elif date_format == DD_YY_MM:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return day + date_delimeter + str(int(year) % 100) + date_delimeter + month
    # MM_DD_YY
    elif date_format == MM_DD_YY:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return month + date_delimeter + day + date_delimeter + str(int(year) % 100)
    # MM_YY_DD
    elif date_format == MM_YY_DD:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return month + date_delimeter + str(int(year) % 100) + date_delimeter + day
    # YY_MM_DD
    elif date_format == YY_MM_DD:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return str(int(year) % 100) + date_delimeter + month + date_delimeter + day
    # YY_DD_MM
    elif date_format == YY_DD_MM:
        day, month = change_day_month_format(day=day, month=month, date_format=DOUBLE)
        return str(int(year) % 100) + date_delimeter + day + date_delimeter + month

    # D_M_YY
    elif date_format == D_M_YY:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return day + date_delimeter + month + date_delimeter + str(int(year) % 100)
    # D_YY_M
    elif date_format == D_YY_M:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return day + date_delimeter + str(int(year) % 100) + date_delimeter + month
    # M_D_YY
    elif date_format == M_D_YY:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return month + date_delimeter + day + date_delimeter + str(int(year) % 100)
    # M_YY_D
    elif date_format == M_YY_D:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return month + date_delimeter + str(int(year) % 100) + date_delimeter + day
    # YY_M_D
    elif date_format == YY_M_D:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return str(int(year) % 100) + date_delimeter + month + date_delimeter + day
    # YY_D_M
    elif date_format == YY_D_M:
        day, month = change_day_month_format(day=day, month=month, date_format=SINGLE)
        return str(int(year) % 100) + date_delimeter + day + date_delimeter + month


def change_date_format_in_list(list_event: [], date_index, date_format_from: str, date_format_to: str,
                               date_delimeter_from=del_slash, date_delimeter_to=del_slash, century=21):
    list_date_new = []  # list for new dates
    for event in list_event:  # parsing all events in list
        day, month, year = date_break_to_day_month_year(date=event[date_index], date_format=date_format_from,
                                                        date_delimeter=date_delimeter_from, century=century)

        date_tmp = set_date_format(day=day, month=month, year=year,
                                   date_format=date_format_to, date_delimeter=date_delimeter_to)
        event[date_index] = date_tmp
        list_date_new.append(event)

    return list_date_new


def change_date_format_from_string(str_date: str, date_format_from: str, date_format_to: str,
                                   date_delimeter_from=del_slash, date_delimeter_to=del_slash, century=21):
    day, month, year = date_break_to_day_month_year(date=str_date, date_format=date_format_from,
                                                    date_delimeter=date_delimeter_from, century=century)
    new_date = set_date_format(day=day, month=month, year=year,
                               date_format=date_format_to, date_delimeter=date_delimeter_to)
    return new_date

# ------------------------------------- #
# ---------- 3) GENERAL USED ---------- #
# ------------------------------------- #


def isLeap(year=dt.datetime.now().year):
    """
    :param year: The year we need to check
    :return: True/False
    """
    if year % 4 == 0:  # if is integer divisible by 4
        if year % 100 != 0 or year % 400 == 0:  # if is not integer divisible by 100 or is integer divisible by 400
            return True  # is leap
        else:
            return False  # is not leap
    else:
        return False  # is not leap


def break_date_range_to_periods(date_start, date_end, period_step, date_format, date_delimeter=del_slash, century=21):
    day_start, month_start, year_start = date_break_to_day_month_year(date=date_start, date_format=date_format,
                                                                      date_delimeter=date_delimeter, century=century)

    date_range_list = []
    curr_day = int(day_start)
    curr_month = int(month_start)
    curr_year = int(year_start)
    while True:
        next_day = curr_day + period_step
        next_month = curr_month
        next_year = curr_year

        month_max_date = list_int_month_days_not_leap[curr_month - 1]
        if isLeap(curr_year):
            month_max_date = list_int_month_days_leap[curr_month - 1]

        if next_day > month_max_date:
            next_day = next_day % month_max_date
            next_month += 1
            if next_month > 12:
                next_month = next_month % 12
                next_year += 1

        curr_date = set_date_format(day=str(curr_day), month=str(curr_month), year=str(curr_year),
                                    date_format=date_format, date_delimeter=date_delimeter)
        next_date = set_date_format(day=str(next_day), month=str(next_month), year=str(next_year),
                                    date_format=date_format, date_delimeter=date_delimeter)

        if next_date > date_end:
            break
        date_range_list.append([curr_date, next_date])

        curr_day = next_day + 1
        curr_month = next_month
        curr_year = next_year

        if curr_day > month_max_date:
            curr_day = curr_day % month_max_date
            curr_month += 1
            if curr_month > 12:
                curr_month = next_month % 12
                curr_year += 1

    return date_range_list


def create_string_list_date_range(list_input: [], del_input=del_slash, del_output=del_slash):
    output_list = []
    for date_range in list_input:
        start_date = date_range[0]
        end_date = date_range[1]
        output_list.append(start_date.replace(del_input, del_output) + "_" + end_date.replace(del_input, del_output))
    return output_list


def merge_values_in_date_range_list(list_input: [], date_index: int, date_range_list: [], merge_type=merge_mean,
                                    del_input=del_slash, del_output=del_slash, del_use=False):
    import statistics

    def raw_data(in_row):
        tmp_in_raw_data = []
        for raw_index in range(0, len(in_row)):
            if raw_index != date_index:
                tmp_in_raw_data.append(in_row[raw_index])
        return tmp_in_raw_data

    merged_list = []  # Create an empty list
    tmp_zero_list = []  # create a list with size of data_length - 1 (-1 due to one column is the date column)
    tmp_list_of_empty_lists = []  # create a list with size of data_length - 1 (-1 due to one column is the date column)
    for _ in range(len(list_input[0]) - 1):  # append the values
        tmp_zero_list.append(0.0)
        tmp_list_of_empty_lists.append([])

    for date_range in date_range_list:  # for each date range
        start_date = date_range[0]  # set the start date
        end_date = date_range[1]  # set the end date
        str_date_range = start_date + "_" + end_date  # create a sting with date range
        if del_use:  # if use del then create a sting with date range using the del_input and del_output
            str_date_range = start_date.replace(del_input, del_output) + "_" + end_date.replace(del_input, del_output)

        if merge_type == merge_min:  # if merge type is min
            tmp_data_min = []  # temporaty data list to store the min values
            for row in list_input:  # for row in list_input
                if start_date <= row[date_index] <= end_date:  # if date in date range
                    if start_date == row[date_index]:  # if start date equals to row_date
                        tmp_data_min = raw_data(in_row=row)  # create the temporary list
                    else:  # else
                        tmp_raw_data = raw_data(in_row=row)  # create a temporary list to store row date
                        for index in range(0, len(tmp_raw_data)):  # for index in range
                            if tmp_raw_data[index] < tmp_data_min[index]:  # check if the new value is minimum
                                tmp_data_min[index] = tmp_raw_data[index]  # set the value
            merged_list.append([str_date_range, tmp_data_min])

        elif merge_type == merge_max:  # if merge type is max
            tmp_data_max = []  # temporaty data list to store the max values
            for row in list_input:  # for row in list_input
                if start_date <= row[date_index] <= end_date:  # if date in date range
                    if start_date == row[date_index]:  # if start date equals to row_date
                        tmp_data_max = raw_data(in_row=row)  # create the temporary list
                    else:  # else
                        tmp_raw_data = raw_data(in_row=row)  # create a temporary list to store row date
                        for index in range(0, len(tmp_raw_data)):  # for index in range
                            if tmp_raw_data[index] > tmp_data_max[index]:  # check if the new value is minimum
                                tmp_data_max[index] = tmp_raw_data[index]  # set the value
            merged_list.append([str_date_range, tmp_data_max])

        elif merge_type == merge_mean:  # if merge type is mean
            tmp_data_mean = tmp_zero_list.copy()  # temporaty data list to calculate the mean values
            mean_counter = 0
            for row in list_input:  # for row in list_input
                if start_date <= row[date_index] <= end_date:  # if date in date range
                    mean_counter += 1
                    tmp_raw_data = raw_data(in_row=row)  # create a temporary list to store row date
                    for index in range(0, len(tmp_raw_data)):  # for index in range
                        tmp_data_mean[index] += tmp_raw_data[index]  # add the values
                    # print(start_date, row[date_index], end_date, tmp_data_mean)
            tmp_out_mean = [round(x / mean_counter, 1) for x in tmp_data_mean]
            merged_list.append([str_date_range, tmp_out_mean])

        elif merge_type == merge_median:  # if merge type is median
            tmp_data_median = tmp_list_of_empty_lists.copy()  # temporaty data list to calculate the median values
            for row in list_input:  # for row in list_input
                if start_date <= row[date_index] <= end_date:  # if date in date range
                    tmp_raw_data = raw_data(in_row=row)  # create a temporary list to store row date
                    for index in range(0, len(tmp_raw_data)):  # for index in range
                        tmp_data_median[index].append(tmp_raw_data[index])  # add the values
            for index in range(0, len(tmp_data_median)):
                tmp_data_median[index] = statistics.median(tmp_data_median[index])
            merged_list.append([str_date_range, tmp_data_median])

    return merged_list


class MyCalendar:
    def __init__(self, list_of_years: [] = None, is_time: bool = False, date_format=DD_MM_YYYY,
                 date_delimeter=del_slash, time_format=HH_MM, time_delimeter=del_colon,
                 hour_start=0, hour_end=24, hour_step=1, minute_start=0, minute_end=60, minute_step=0):
        if list_of_years is None or list_of_years == EMPTY_LIST:  # if list_of_year is None
            list_of_years = [dt.datetime.now().year]  # create a list with only the current year
        self.list_of_years = list_of_years  # set the self.list_of_years
        self.list_of_year_is_leap = [isLeap(year) for year in list_of_years]  # set list_of_is_leap
        self.date_format = date_format  # set the date format
        self.date_delimeter = date_delimeter  # set the date delimeter

        self.is_time = is_time  # set the self.is_time
        self.time_format = time_format  # set the time format
        self.time_delimeter = time_delimeter  # set the time delimeter
        self.hour_start = hour_start  # set hour start
        self.hour_end = hour_end  # set hour end
        self.hour_step = hour_step  # set hour step
        self.minute_start = minute_start  # set minute start
        self.minute_end = minute_end  # set minute end
        self.minute_step = minute_step  # set minute step
        self.list_timespamp = None  # set the list timestamp
        self.int_list_size = 0  # set the list size to 0

        self.dict_calendar = EMPTY_DICT.copy()  # create the calendar
        self.create()
        self.event_names = []
        self.header_names = []

    def change_hour_number_in_a_cal_list_to_time(self, list_event: [], time_index):
        """
        In a given list change the time forma to calendar time format
        :param list_event: the list of events
        :param time_index: the index of time column in list
        :return: list_hour
        """
        list_hour_tmp = []
        if self.time_format == HH_MM:
            for event in list_event:
                event[time_index] = '%02i' % int(event[time_index]) + ':00'
                list_hour_tmp.append(event)
        elif self.time_format == HH_MM_SS:
            for event in list_event:
                event[time_index] = '%02i' % int(event[time_index]) + ':00'
                list_hour_tmp.append(event)
        return list_hour_tmp

    def change_date_format_in_a_cal_list(self, list_event: [], date_index, date_format_from: str,
                                         date_delimeter_from=del_slash, century=21):
        list_date_new = []  # list for new dates
        for event in list_event:  # parsing all events in list
            day, month, year = date_break_to_day_month_year(date=event[date_index], date_format=date_format_from,
                                                            date_delimeter=date_delimeter_from, century=century)

            date_tmp = set_date_format(day=day, month=month, year=year,
                                       date_format=self.date_format, date_delimeter=self.date_delimeter)
            event[date_index] = date_tmp
            list_date_new.append(event)

        return list_date_new

    def print(self, print_number=None):
        if print_number is None:
            for date_keys in self.dict_calendar.keys():
                print(date_keys, self.dict_calendar[date_keys])
        else:
            break_index = 0
            for date_keys in self.dict_calendar.keys():
                if break_index == print_number:
                    break
                print(date_keys, self.dict_calendar[date_keys])
                break_index += 1

    def print_date(self):
        for date_keys in self.dict_calendar.keys():
            print(date_keys)

    def create(self):
        """
        Calendar Creation.
        :return: Nothing
        """
        list_timestamp, int_list_size = set_time(hour_start=self.hour_start,
                                                 hour_end=self.hour_end,
                                                 hour_step=self.hour_step,
                                                 minute_start=self.minute_start,
                                                 minute_end=self.minute_end,
                                                 minute_step=self.minute_step)

        self.list_timespamp = list_timestamp
        self.int_list_size = int_list_size

        for year_index in range(len(self.list_of_years)):
            if self.list_of_year_is_leap[year_index]:
                for month in range(0, 12):
                    for day in range(0, list_int_month_days_leap[month]):
                        date_tmp = set_date_format(day=list_str_id_days_dd[day],
                                                   month=list_str_id_months_mm[month],
                                                   year=str(self.list_of_years[year_index]),
                                                   date_format=self.date_format,
                                                   date_delimeter=self.date_delimeter)
                        self.dict_calendar[date_tmp] = EMPTY_DICT.copy()
                        if self.is_time:
                            for timestamp in self.list_timespamp:
                                self.dict_calendar[date_tmp][timestamp] = EMPTY_DICT.copy()
            else:
                for month in range(0, 12):
                    for day in range(0, list_int_month_days_not_leap[month]):
                        date_tmp = set_date_format(day=list_str_id_days_dd[day],
                                                   month=list_str_id_months_mm[month],
                                                   year=str(self.list_of_years[year_index]),
                                                   date_format=self.date_format,
                                                   date_delimeter=self.date_delimeter)
                        self.dict_calendar[date_tmp] = EMPTY_DICT.copy()
                        if self.is_time:
                            for timestamp in self.list_timespamp:
                                self.dict_calendar[date_tmp][timestamp] = EMPTY_DICT.copy()

    def add_list_key_event_to_calendar(self, list_key_event: [], list_of_headers=None):
        for date in self.dict_calendar.keys():
            if self.is_time:
                for time in self.dict_calendar[date].keys():
                    if list_of_headers is None:
                        for event in list_key_event:
                            self.dict_calendar[date][time][event] = NaN
                    else:
                        for event in list_key_event:
                            self.dict_calendar[date][time][event] = {}
                            for header in list_of_headers:
                                self.dict_calendar[date][time][event][header] = NaN
            else:
                if list_of_headers is None:
                    for event in list_key_event:
                        self.dict_calendar[date][event] = NaN
                else:
                    for event in list_key_event:
                        self.dict_calendar[date][event] = {}
                        for header in list_of_headers:
                            self.dict_calendar[date][event][header] = NaN

    def add_events_to_calendar(self, list_of_events: [], date_index, time_index, first_row_header: bool,
                               list_of_headers=None, no_data_value=NaN, event_index=None, input_event_name=None):
        self.event_names = []
        self.header_names = []
        start_index = 0
        if first_row_header:
            start_index = 1
            header_size = len(list_of_events[0])
        else:
            header_size = len(list_of_headers)

        for i in range(start_index, len(list_of_events)):
            date = list_of_events[i][date_index]
            if self.is_time:
                time = list_of_events[i][time_index]
                if first_row_header:
                    event_name = list_of_events[i][event_index]
                    if event_name not in self.event_names:
                        self.event_names.append(event_name)
                    for j in range(0, header_size):
                        header_name = list_of_events[0][j]
                        if header_name not in self.header_names:
                            self.header_names.append(header_name)
                        if event_name not in self.dict_calendar[date].keys():
                            self.dict_calendar[date][time_index][event_name] = {}
                        if list_of_events[i][j] is '':
                            self.dict_calendar[date][time_index][event_name][header_name] = no_data_value
                        else:
                            self.dict_calendar[date][time_index][event_name][header_name] = list_of_events[i][j]
                else:
                    event_name = input_event_name
                    for j in range(0, header_size):
                        header_name = list_of_headers[j]
                        if header_name not in self.header_names:
                            self.header_names.append(header_name)
                        if event_name not in self.dict_calendar[date][time].keys():
                            self.dict_calendar[date][time][event_name] = {}
                        if list_of_events[i][j] is '':
                            self.dict_calendar[date][time][event_name][header_name] = no_data_value
                        else:
                            self.dict_calendar[date][time][event_name][header_name] = list_of_events[i][j]

            else:
                # print(i, date, list_of_events[i][event_index], list_of_events[i])
                if first_row_header:
                    event_name = list_of_events[i][event_index]
                    if event_name not in self.event_names:
                        self.event_names.append(event_name)
                    for j in range(0, header_size):
                        header_name = list_of_events[0][j]
                        if header_name not in self.header_names:
                            self.header_names.append(header_name)
                        if event_name not in self.dict_calendar[date].keys():
                            self.dict_calendar[date][event_name] = {}
                        if list_of_events[i][j] is '':
                            self.dict_calendar[date][event_name][header_name] = no_data_value
                        else:
                            self.dict_calendar[date][event_name][header_name] = list_of_events[i][j]
                else:
                    pass

    def fill_calendar_with_missing_keys(self, no_data_value=NaN):
        for date in self.dict_calendar.keys():
            if self.is_time:
                for time in self.dict_calendar[date].keys():
                    for event in self.event_names:
                        if event not in self.dict_calendar[date][time].keys():
                            self.dict_calendar[date][time][event] = {}
                            for header in self.header_names:
                                self.dict_calendar[date][event][time][header] = no_data_value
                        else:
                            for header in self.header_names:
                                if header not in self.dict_calendar[date][time][event].keys():
                                    self.dict_calendar[date][time][event][header] = no_data_value
            else:
                for event in self.event_names:
                    if event not in self.dict_calendar[date].keys():
                        self.dict_calendar[date][event] = {}
                        for header in self.header_names:
                            self.dict_calendar[date][event][header] = no_data_value
                    else:
                        for header in self.header_names:
                            if header not in self.dict_calendar[date][event].keys():
                                self.dict_calendar[date][event][header] = no_data_value

    def dict_to_list(self, date_range=None):
        header_list_tmp = []
        output_list = []
        append_header = True
        is_data_in_range = False
        can_append = True

        for date in self.dict_calendar.keys():  # for each date
            if self.is_time:  # If time exists
                for time in self.dict_calendar[date].keys():  # for each time
                    if type(self.dict_calendar[date][time]) is dict:  # if is dict
                        for event in self.dict_calendar[date][time].keys():  # for each event
                            if type(self.dict_calendar[date][time][event]) is dict:  # if its dict
                                tmp_list = []
                                for header in self.dict_calendar[date][time][event].keys():  # for each header
                                    if header not in header_list_tmp:  # if header not in list
                                        header_list_tmp.append(header)  # append it
                                    if date_range is None:
                                        tmp_list.append(self.dict_calendar[date][time][event][header])
                                    else:
                                        if date_range[0] == date or is_data_in_range:
                                            is_data_in_range = True
                                            tmp_list.append(self.dict_calendar[date][time][event][header])
                                            if date_range[1] == date:
                                                is_data_in_range = False
                                                can_append = False
                                if append_header:
                                    output_list.append(header_list_tmp)
                                    append_header = False
                                if tmp_list and can_append:
                                    output_list.append(tmp_list)
                            else:
                                tmp_list = []
                                for event_name in self.dict_calendar[date][time]:
                                    if event_name not in header_list_tmp:
                                        header_list_tmp.append(event_name)
                                    if date_range is None:
                                        tmp_list.append(self.dict_calendar[date][time][event_name])
                                    else:
                                        if date_range[0] == date or is_data_in_range:
                                            is_data_in_range = True
                                            tmp_list.append(self.dict_calendar[date][time][event_name])
                                            if date_range[1] == date:
                                                is_data_in_range = False
                                                can_append = False
                                if append_header:
                                    output_list.append(header_list_tmp)
                                    append_header = False
                                if tmp_list and can_append:
                                    output_list.append(tmp_list)
                    else:
                        pass
            else:
                if type(self.dict_calendar[date]) is dict:
                    for event in self.dict_calendar[date].keys():
                        if type(self.dict_calendar[date][event]) is dict:
                            tmp_list = []
                            for header in self.dict_calendar[date][event].keys():
                                if header not in header_list_tmp:
                                    header_list_tmp.append(header)
                                tmp_list.append(self.dict_calendar[date][event][header])
                            if append_header:
                                output_list.append(header_list_tmp)
                                append_header = False
                            output_list.append(tmp_list)
                        else:
                            tmp_list = []
                            for event_name in self.dict_calendar[date]:
                                if event_name not in header_list_tmp:
                                    header_list_tmp.append(event_name)
                                if date_range is None:
                                    tmp_list.append(self.dict_calendar[date][event_name])
                                else:
                                    if date_range[0] == date or is_data_in_range:
                                        is_data_in_range = True
                                        tmp_list.append(self.dict_calendar[date][event_name])
                                        if date_range[1] == date:
                                            is_data_in_range = False

                            if append_header:
                                output_list.append(header_list_tmp)
                                append_header = False
                            output_list.append(tmp_list)
                    else:
                        pass
        return output_list
