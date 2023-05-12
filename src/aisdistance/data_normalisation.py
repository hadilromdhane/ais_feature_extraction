# In this file we will  calculate the TCPA and DCPA
# We will also do some data normalisation so that all the values we have fixed units
# NM for distances - radians for angles - knots for speed

# Import necessary libraries
import pandas as pd
import aisdistance.utm_module as utm_module


# Import our datafarme ( the output of goal_generation)
final_df = pd.read_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_folder/missing_ts_added.csv')


for i in range(len(final_df)):
    E0, N0, zone_numbr0, zone_letter0 = utm_module.from_latlon(
        final_df.loc[i, 'OS_latitude'], final_df.loc[i, 'OS_longitude'], None, None)
    heading0 = final_df.loc[i, 'OS_heading']
    course0 = final_df.loc[i, 'OS_course']
    speed0 = final_df.loc[i, 'OS_speed']

    E1, N1, zone_numbr1, zone_letter1 = utm_module.from_latlon(
        final_df.loc[i, 'TS_latitude'],  final_df.loc[i, 'TS_longitude'], None, None)
    heading1 = final_df.loc[i, 'TS_heading']
    course1 = final_df.loc[i, 'TS_course']
    speed1 = final_df.loc[i, 'TS_speed']

    final_df.loc[i, 'OS_distance_TS'] = utm_module.meter_to_NM(
        utm_module.ED(N0, E0, N1, E1))
    final_df.loc[i, 'OS_rb_TS'] = utm_module.bng_rel(N0, E0, N1, E1, heading0)
    final_df.loc[i, 'TCPA_TS_OS'] = utm_module.tcpa(
        N0, E0, N1, E1, utm_module.dtr(course0), utm_module.dtr(course1), speed0, speed1)
    final_df.loc[i, 'DCPA_TS_OS'] = utm_module.cpa(
        N0, E0, N1, E1,  utm_module.dtr(course0),  utm_module.dtr(course1), speed0, speed1)[0]


# normalise the data
final_df['OS_course'] = utm_module.dtr(final_df['OS_course'])
final_df['TS_course'] = utm_module.dtr(final_df['TS_course'])

final_df['OS_heading'] = utm_module.dtr(final_df['OS_heading'])
final_df['TS_heading'] = utm_module.dtr(final_df['TS_heading'])

final_df['OS_distance_goal'] = utm_module.meter_to_NM(
    final_df['OS_distance_goal'])
final_df['DCPA_TS_OS'] = utm_module.meter_to_NM(final_df['DCPA_TS_OS'])


final_df = final_df.round(6)
final_df.to_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_folder/final_data.csv', index=False)
