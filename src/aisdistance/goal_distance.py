# This file's purpose is to extract the exact distance and relative bearing to the goal.
# The goal of an OS is defined as a ship that have the MMSI with a distance between 10NM and 15NM from the current position.


# Import necessary libraries
import pandas as pd
import aisdistance.utm_module as utm_module

# Load data from file(the intial decoded data)
df = pd.read_csv('2020_07_01_sample_1000.csv')


# Keep only relevant columns for goal generation phase
df = df[['timestamp', 'latitude', 'longitude',
         'MMSI', 'heading', 'speed', 'course']]

# Remove invalid heading values and duplicates
# df = df[df['heading'] != 511]
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.drop_duplicates(subset=['timestamp', 'MMSI'], keep=False, inplace=True)
df = df.reset_index(drop=True)


# Initialize counters and an empty DataFrame for storing results
i = 0
k = 0
data = pd.DataFrame()

# Loop over all rows in the data to find ships that meet criteria
while i < len(df):
    # Select only ships with the same MMSI as the current row
    df_distance = df[df['MMSI'] == df.loc[i, 'MMSI']]
    df_distance = df_distance.reset_index(drop=True)
    j = 0
    while j < len(df_distance):
        # Calculate the distance to the potential goals
        E0, N0, zone_numbr0, zone_letter0 = utm_module.from_latlon(df.loc[i, 'latitude'], df.loc[i, 'longitude'], utm_module.latlon_to_zone_number(
            df.loc[i, 'latitude'], df.loc[i, 'longitude']), utm_module.latitude_to_zone_letter(df.loc[i, 'latitude']))
        E1, N1, zone_numbr1, zone_letter1 = utm_module.from_latlon(df_distance.loc[j, 'latitude'], df_distance.loc[j, 'longitude'], utm_module.latlon_to_zone_number(
            df_distance.loc[j, 'latitude'], df_distance.loc[j, 'longitude']), utm_module.latitude_to_zone_letter(df_distance.loc[j, 'latitude']))
        heading0 = df.loc[i, 'heading']
        distance_goal = utm_module.ED(N0, E0, N1, E1)
        distance_goal = utm_module.meter_to_NM(distance_goal)

        # Check if the distance between 10 and 15 nautical miles, if yes so this the goal we are seaching for
        if (10 <= distance_goal <= 15):

            data = pd.concat([data, df.iloc[[i]]], ignore_index=True)
            data.loc[k, 'distance_goal'] = distance_goal

            # Calculate the realtive bearing to the goal and add it as column for the output dataframe
            rb_goal = utm_module.bng_rel(N0, E0, N1, E1, heading0)
            data.loc[k, 'rb_goal'] = rb_goal

            k = k+1
            break

        else:
            j = j+1

    i += 1

data.to_csv('results.csv')
