# In this file, we will add any missing target ships in cases where the number of existing target ships is less than 4
# The added target ships will be non-collision risk  ships


# Import necessary libraries
import pandas as pd
import aisdistance.utm_module as utm_module
import math


# Import our datafarme
target_ships = pd.read_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_folder/ts_extracted.csv')


# Function to calculate coordinates for a new target ship
def find_TS_coordinates(p1_north, p1_east, p1_heading):
    distance = 5556
    # the bearing angle between TS and OS
    bearing = p1_heading+180

    # Convert the bearing to radians
    bearing_rad = math.radians(bearing)

    # Find the x and y displacements using trigonometry
    north_disp = distance * math.cos(bearing_rad)
    east_disp = distance * math.sin(bearing_rad)

    # Calculate the x and y coordinates of P2
    p2_north = p1_north + north_disp
    p2_east = p1_east + east_disp
    p2_heading = p1_heading+180

    return p2_north, p2_east, p2_heading


# Function to add non-collision risk target ships to a dataframe
def add_TS(OS_timestamp, OS_MMSI, OS_lat, OS_long, OS_heading, OS_course, OS_speed, OS_distance_goal, OS_rb_goal, number_TS_to_add):

    # Create a list of DataFrames to concatenate
    new_rows = []
    E0, N0, zone_numbr0, zone_letter0 = utm_module.from_latlon(
        OS_lat, OS_long, utm_module.latlon_to_zone_number(OS_lat, OS_long), utm_module.latitude_to_zone_letter(OS_lat))

    N1, E1, heading1 = find_TS_coordinates(N0, E0, OS_heading)

    TS_lat, TS_lon = utm_module.to_latlon(E1, N1, zone_numbr0, zone_letter0)

    for _ in range(number_TS_to_add):
        row = {
            'timestamp': OS_timestamp,
            'OS_MMSI': OS_MMSI,
            'OS_latitude': OS_lat,
            'OS_longitude': OS_long,
            'OS_heading': OS_heading,
            'OS_course': OS_course,
            'OS_speed': OS_speed,
            'OS_distance_goal': OS_distance_goal,
            'OS_rb_goal': OS_rb_goal,



            # feature of the added target ship
            'TS_latitude': TS_lat,
            'TS_longitude': TS_lon,
            'TS_heading': heading1,
            'TS_course': OS_course+180,
            'TS_speed': 0
        }
        new_rows.append(pd.DataFrame(row, index=[0]))

    return pd.concat(new_rows, ignore_index=True)


i = 0
concatenated_df = pd.DataFrame()  # create an empty DataFrame
while i < len(target_ships):

    first_df = target_ships[(target_ships.OS_MMSI == target_ships.loc[i, 'OS_MMSI']) &
                            (target_ships.timestamp ==
                             target_ships.loc[i, 'timestamp'])
                            ]

    # Create a dictionary with data for the new target ship
    OS_timestamp = target_ships.loc[i, 'timestamp']
    OS_MMSI = target_ships.loc[i, 'OS_MMSI']
    OS_lat = target_ships.loc[i, 'OS_latitude']
    OS_long = target_ships.loc[i, 'OS_longitude']
    OS_heading = target_ships.loc[i, 'OS_heading']
    OS_speed = target_ships.loc[i, 'OS_speed']
    OS_course = target_ships.loc[i, 'OS_course']
    OS_distance_goal = target_ships.loc[i, 'OS_distance_goal']
    OS_rb_goal = target_ships.loc[i, 'OS_rb_goal']

    # this nb value will added to the columns that don't really have any target ship so +nb TS ships will be added
    nb = len(first_df[first_df['TS_longitude'].isna()])

    if len(first_df) == 0:
        new_TS = add_TS(OS_timestamp, OS_MMSI, OS_lat, OS_long, OS_heading,
                        OS_course, OS_speed, OS_distance_goal, OS_rb_goal, 4+nb)

    elif len(first_df) == 1:
        new_TS = add_TS(OS_timestamp, OS_MMSI, OS_lat, OS_long, OS_heading,
                        OS_course, OS_speed, OS_distance_goal, OS_rb_goal, 3+nb)

    elif len(first_df) == 2:
        new_TS = add_TS(OS_timestamp, OS_MMSI, OS_lat, OS_long, OS_heading,
                        OS_course, OS_speed, OS_distance_goal, OS_rb_goal, 2+nb)

    elif len(first_df) == 3:
        new_TS = add_TS(OS_timestamp, OS_MMSI, OS_lat, OS_long, OS_heading,
                        OS_course, OS_speed, OS_distance_goal, OS_rb_goal, 1+nb)

    i += len(first_df)
    # append current iteration's result to concatenated_df
    concatenated_df = pd.concat(
        [concatenated_df, first_df, new_TS], ignore_index=True)


final_df = concatenated_df
final_df = final_df.dropna()
final_df = final_df.reset_index(drop=True)


final_df.to_csv(
    'C:/Users/msi/Desktop/Project_PFE/data_folder/missing_ts_added.csv', index=False)
