import glob
import os

import pandas as pd


def clean_gps(**config) -> None:
    """Clean raw GPS data into a csv format that has just lat, on, and alt values

    clean_gps finds all logs inside of "raw" folder and creates csv of cleaned 
    data to gps_data under a folder with the same name as it is in "*/raw".
    """
    all_raw_folders = glob.glob(os.path.join(config["gps_raw_data"], '*'))
    for folder_path in all_raw_folders:
        if "/0ft" in folder_path:
            print("-" * 80)
            clean_combine_fixed(folder_path)
        # create gps_data folder if it doesn't  exist
        folder_out = folder_path.replace("raw", "gps_data")
        if not os.path.isdir(folder_out):
            os.makedirs(folder_out, exist_ok=True)

        # clean files inside batch and send to folder under same batch name in gps_dats
        all_raw_files = glob.glob(folder_path+ "/*.log")
        for log_file in all_raw_files:
            print("Cleaning raw data at", log_file)
            df_gps = pd.read_csv(log_file, index_col=None, header=None,
                                 names=['lat', 'lon', 'alt'])
            df_gps.lat = df_gps.lat.str.replace("lat=", "").astype(float)
            df_gps.lon = df_gps.lon.str.replace("lon=", "").astype(float)
            df_gps.alt = df_gps.alt.str.replace("alt=", "").astype(float)

            # Update cleaned csv path
            cleaned_file_path = log_file \
                .replace(".log", "_cleaned.csv") \
                .replace("raw", "gps_data")
            df_gps.to_csv(cleaned_file_path, index=False)

        print("Cleaned csv for all raw data inside", folder_path, "at",  folder_out)


def clean_combine_fixed(folder_path) -> None:
    """Cleans every file within the fixed folder and combines all of the cleaned 
    files into one csv file.
    """
    print("Combining gps data")
    # make a new folder inside temp to hold cleaned data for each batch
    all_raw_files = glob.glob(folder_path + "/*.log")
    all_df = df = pd.DataFrame(columns = ['lat', 'lon', 'alt'])
    
    # clean files inside batch and send to folder under same batch name in gps_dats
    for log_file in all_raw_files:
        print("Combining raw data at", log_file)
        df_x = pd.read_csv(log_file, index_col=None, header=None, names=['lat', 'lon', 'alt'])
        df_x.lat = df_x.lat.str.replace("lat=", "").astype(float)
        df_x.lon = df_x.lon.str.replace("lon=", "").astype(float)
        df_x.alt = df_x.alt.str.replace("alt=", "").astype(float)
        all_df = all_df.append(df_x, ignore_index=True)
        
    folder_out = "data/CEP/"
    if not os.path.isdir(folder_out):
        os.makedirs(folder_out, exist_ok=True)

    all_df_path = folder_out + "fixed_all.csv"
    all_df.to_csv(all_df_path, index=False)
