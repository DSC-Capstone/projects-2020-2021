import os

from bokeh.plotting import gmap
from bokeh.models import GMapOptions, ColumnDataSource
from bokeh.io import output_file, show, save
import pandas as pd


def visualize(data_path, vis_path) -> None:
    """Create google maps render of path that is reported from gps"""
    df_gps = pd.read_csv(data_path)
    lats = df_gps.lat.values
    lons = df_gps.lon.values

    api_key = "AIzaSyAx6D3SIFzHvW4nDrNjvxXCd4KZRVxlHnk"
    #api_key = os.getenv("GMAP_API_KEY")

    # configuring the Google map
    google_map_options = GMapOptions(lat=lats[0],
                                     lng=lons[0],
                                     map_type="satellite",
                                     zoom=25)

    # generating the Google map
    google_map = gmap(api_key, google_map_options)
    google_map.line(lons, lats, color="red")

    # save html file
    output_file(vis_path)
    save(google_map)


def visualize_all(**config) -> None:

    for folder in config["data_folders"]:
        vis_folder = folder.replace("data", "vis")

        if not os.path.exists(vis_folder):
            os.makedirs(vis_folder)

        for gps_csv in os.listdir(folder):
            data_path = folder + gps_csv
            vis_path = vis_folder + gps_csv.replace("csv", "html")
            visualize(data_path, vis_path)
