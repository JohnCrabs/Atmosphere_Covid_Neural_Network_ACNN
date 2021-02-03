import re
import ee
from my_lib import my_calendar_v2 as my_cal_v2
import time

ee.Initialize()


def clear_tasks():
    operation_list = ee.batch.data.listOperations()
    # print(operation_list)
    index = 1
    list_len = len(operation_list)
    for operation in operation_list:
        print("(" + str(index) + "/" + str(list_len) + ") Cancel operation: " + operation['name'])
        ee.batch.data.cancelOperation(operation['name'])
        index += 1


def download_image_from_collection(collection_id, image_band, img_name, list_date_range, list_countries, scale=5000,
                                   waiting_time=None):
    country_collection = ee.FeatureCollection('users/midekisa/Countries')  # add countries boundary geometries
    for country in list_countries:
        print("\nCreate boundary geometry for  " + country)
        country_geometry = country_collection.filter(ee.Filter.eq('Country', country))
        country_name_for_path = re.sub(r"[^\w\s]", '', country).replace(" ", "_")
        # print(country_name_for_path)

        for date_range in list_date_range:
            start_date = date_range[0]
            end_date = date_range[1]
            print("\t* Prepare collection for period (" + start_date + " to " + end_date + "):")

            collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(
                country_geometry)
            image = collection.mean().clip(country_geometry)
            image = image.select([image_band])

            start_date_for_path = my_cal_v2.change_date_format_from_string(str_date=start_date,
                                                                           date_format_from=my_cal_v2.YYYY_MM_DD,
                                                                           date_format_to=my_cal_v2.YYYY_MM_DD,
                                                                           date_delimeter_from=my_cal_v2.del_dash,
                                                                           date_delimeter_to=my_cal_v2.del_none,
                                                                           century=21)
            end_date_for_path = my_cal_v2.change_date_format_from_string(str_date=end_date,
                                                                         date_format_from=my_cal_v2.YYYY_MM_DD,
                                                                         date_format_to=my_cal_v2.YYYY_MM_DD,
                                                                         date_delimeter_from=my_cal_v2.del_dash,
                                                                         date_delimeter_to=my_cal_v2.del_none,
                                                                         century=21)

            my_description = img_name + '_' + country_name_for_path + '_' + str(scale) + '_' + start_date_for_path + '_' + end_date_for_path
            print("\t\tExport Image via description (task/image name):")
            print("\t\t" + my_description)

            task_bash = {
                'image': image,
                'description': my_description,
                'region': country_geometry.geometry(),
                'scale': scale,
                'folder': 'GEE_' + country_name_for_path
            }
            task = ee.batch.Export.image.toDrive(**task_bash)
            task.start()
            print("Run task on https://code.earthengine.google.com/ task window.")
            is_timer = False
            timer_value = 0
            if waiting_time is not None:
                is_timer = True
                timer_value = waiting_time
            while task.status()['state'] is 'READY':
                if is_timer:
                    time.sleep(1)
                    timer_value -= 1
                    if timer_value == 0:
                        break

            print("Task finished successfully!")
