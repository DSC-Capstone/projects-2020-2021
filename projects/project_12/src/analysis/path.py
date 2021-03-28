import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_path(**config):
    """Take in csv of gps path and clean coordiantes"""
    df_center_line = pd.read_csv(config["csv_path"])
    length = len(df_center_line)/2

    # pretty much random guessing of how to clean path
    lat = df_center_line["latitude(degrees)"][np.arange(int(length), int(length*1.47), 2)].values
    lon = df_center_line["longitude(degrees)"][np.arange(int(length), int(length*1.47), 2)].values

    # connect starting and ending points by adding starting point to end of path
    lat = np.append(lat, lat[0])
    lon = np.append(lon, lon[0])

    fig, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    ax.plot(lat, lon)
    plt.show()

    # Save cleaned path to csv
    pd.DataFrame({"lat": lat, "lon": lon}).to_csv("thunderhill_west_gps_path.csv")


if __name__ == "__main__":
    print("Generating Path")
