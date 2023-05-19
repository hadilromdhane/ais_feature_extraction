# In this file we will extract the initially exciting target ships
# The maximum number of target ships to extarct should be 4
# A target ship is : Ship with a distance of maximum 3NM from the OS

# Import necessary libraries
import pandas as pd
import aisdistance.utm_module as utm_module


# Import our datafarme ( the output of goal_generation)
df = pd.read_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_hpc/goal_generated.csv')


# Define the distance threshold for including target ships
distance_threshold = 3  # nautical miles
number_TS = 4


dtypes_dict = {'timestamp': 'object',
               'OS_MMSI': 'object',
               'OS_latitude': 'float64',
               'OS_longitude': 'float64',
               'OS_heading': 'float64',
               'OS_speed': 'float64',
               'OS_course': 'float64',
               'OS_distance_goal': 'float64',
               'OS_rb_goal': 'float64',
               'TS_latitude': 'float64',
               'TS_longitude': 'float64',
               'TS_heading': 'float64',
               'TS_course': 'float64',
               'TS_speed': 'float64'
               }

target_ships = pd.DataFrame(columns=dtypes_dict.keys()).astype(dtypes_dict)


k = 0
for i in range(len(df)):
    # the final dataframe that  will containt 4 TS for each OS
    target_ships.loc[k, 'timestamp'] = df.loc[i, 'timestamp']
    target_ships.loc[k, 'OS_MMSI'] = df.loc[i, 'MMSI']
    target_ships.loc[k, 'OS_latitude'] = df.loc[i, 'latitude']
    target_ships.loc[k, 'OS_longitude'] = df.loc[i, 'longitude']
    target_ships.loc[k, 'OS_heading'] = df.loc[i, 'heading']
    target_ships.loc[k, 'OS_speed'] = df.loc[i, 'speed']
    target_ships.loc[k, 'OS_course'] = df.loc[i, 'course']
    target_ships.loc[k, 'OS_distance_goal'] = df.loc[i, 'distance_goal']
    target_ships.loc[k, 'OS_rb_goal'] = df.loc[i, 'rb_goal']

    # intialize a dataframe that contains all the rows that have the same timestamp and diffrent MMSI of the OS (potential target ships)
    df1 = df[(df.timestamp == target_ships.loc[k, 'timestamp']) & (
        df.MMSI != target_ships.loc[k, 'OS_MMSI'])].reset_index(drop=True)
    E0, N0, zone_numbr0, zone_letter0 = utm_module.from_latlon(target_ships.loc[k, 'OS_latitude'], target_ships.loc[k, 'OS_longitude'], utm_module.latlon_to_zone_number(
        target_ships.loc[k, 'OS_latitude'], target_ships.loc[k, 'OS_longitude']), utm_module.latitude_to_zone_letter(target_ships.loc[k, 'OS_latitude']))
    # dtermine the number of target ships already found
    nb_ts = target_ships[target_ships.OS_MMSI ==
                         target_ships.loc[k, 'OS_MMSI']].OS_MMSI.count()

    for j in range(len(df1)):

        E1, N1, zone_numbr1, zone_letter1 = utm_module.from_latlon(df1.loc[j, 'latitude'], df1.loc[j, 'longitude'], utm_module.latlon_to_zone_number(
            df1.loc[j, 'latitude'], df1.loc[j, 'longitude']), utm_module.latitude_to_zone_letter(df1.loc[j, 'latitude']))

        # calculate the distance and number of each potential TS to my OS
        dis = utm_module.meter_to_NM(utm_module.ED(N0, E0, N1, E1))
        if (nb_ts > number_TS):
            break
        elif (dis <= distance_threshold) & (nb_ts < number_TS-1):
            target_ships.loc[k, 'TS_latitude'] = df1.loc[j, 'latitude']
            target_ships.loc[k, 'TS_longitude'] = df1.loc[j, 'longitude']
            target_ships.loc[k, 'TS_heading'] = df1.loc[j, 'heading']
            target_ships.loc[k, 'TS_course'] = df1.loc[j, 'course']
            target_ships.loc[k, 'TS_speed'] = df1.loc[j, 'speed']

    k = k+1


target_ships.to_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_hpc/ts_extracted.csv', index=False)
