
import pandas as pd
import aisdistance.utm_module as utm_module

# import the data
df = pd.read_csv('2020_07_01_sample_1000.csv')
df.head()

# we did some data engineering in order to decrease the number of feature
df = df[['timestamp', 'latitude', 'longitude',
         'MMSI', 'heading', 'speed', 'course']]

# heading equal 511 means that the value of heading don't exist
df = df[df['heading'] != 511]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.drop_duplicates(subset=['timestamp', 'MMSI'], keep=False, inplace=True)

df.drop_duplicates(inplace=True)
df = df.reset_index(drop=True)

i = 0
# Initialize a counter for the rows added to the data DataFrame
k = 0
data = pd.DataFrame()  # Initialize an empty DataFrame
while i < len(df):
    # temporary data frame that contain all the ships with same MMSI as df.loc[i,'MMSI']
    df_distance = df[df['MMSI'] == df.loc[i, 'MMSI']]
    df_distance = df_distance.reset_index(drop=True)
    j = 0
    # Initialize a counter for the rows in the data DataFrame
    while j < len(df_distance):
        # calculate the distance to the goal
        E0, N0, zone_numbr0, zone_letter0 = utm_module.from_latlon(df.loc[i, 'latitude'], df.loc[i, 'longitude'], utm_module.latlon_to_zone_number(
            df.loc[i, 'latitude'], df.loc[i, 'longitude']), utm_module.latitude_to_zone_letter(df.loc[i, 'latitude']))
        E1, N1, zone_numbr1, zone_letter1 = utm_module.from_latlon(df_distance.loc[j, 'latitude'], df_distance.loc[j, 'longitude'], utm_module.latlon_to_zone_number(
            df_distance.loc[j, 'latitude'], df_distance.loc[j, 'longitude']), utm_module.latitude_to_zone_letter(df_distance.loc[j, 'latitude']))
        heading0 = df.loc[i, 'heading']

        #! I need to only add ships with distance betwwen 10 and 15 NM
        distance_goal = utm_module.ED(N0, E0, N1, E1)
        distance_goal = utm_module.meter_to_NM(distance_goal)

        # to modify the interval

        if (1 <= distance_goal <= 2):
            data = pd.concat([data, df.iloc[[i]]], ignore_index=True)
            # Use the counter variable k to index the data DataFrame
            data.loc[k, 'distance_goal'] = distance_goal
            # calculate the realtive bearing to the goal
            rb_goal = utm_module.bng_rel(N0, E0, N1, E1, heading0)
            # Use the counter variable k to index the data DataFrame
            data.loc[k, 'rb_goal'] = rb_goal
            # if I found my goal directly move to the next iteration
            k = k+1  # Increment the counter variable k
            break

        else:
            j = j+1

    i += 1
data.to_csv('results.csv')
