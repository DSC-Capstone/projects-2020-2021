import math
import pandas as pd 
import glob
import os

def get_x(ellps, lat, lon, h) -> float:
    """Transforms latitude and longitude into a cartesian
    x position.
    
    Source
    Author: YeO
    Date: 6/16/2018
    Availability: https://codereview.stackexchange.com/questions/195933/convert-geodetic-coordinates-to-geocentric-cartesian
    """
    wgs84 = (6378137, 298.257223563)
    a, rf = ellps
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    N = a / math.sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (math.sin(lat_rad)) ** 2)
    X = (N + h) * math.cos(lat_rad) * math.cos(lon_rad)
    return X

def get_y(ellps, lat, lon, h) -> float:
    """Transforms latitude and longitude into a cartesian
    y position.
    
    Source
    Author: YeO
    Date: 6/16/2018
    Availability: https://codereview.stackexchange.com/questions/195933/convert-geodetic-coordinates-to-geocentric-cartesian
    """
    a, rf = ellps
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    N = a / math.sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (math.sin(lat_rad)) ** 2)
    Y = (N + h) * math.cos(lat_rad) * math.sin(lon_rad)
    return Y
    
def cartesian_conversion(path) -> pd.DataFrame():
    """ Creates a dataframe and translates the 
    gps coordines into cartesian x and y coordinates
    """
    df = pd.read_csv(path)
    wgs84 = (6378137, 298.257223563)
    df['x'] = df.apply(lambda row: get_x(wgs84, row.lat, row.lon, row.alt), axis =1)
    df['y'] = df.apply(lambda row: get_y(wgs84, row.lat, row.lon, row.alt), axis =1)
    return df


def transform_fixed_cartesian() -> None:
    """ Takes in fixed_all.csv and addss two new columnns:
    the translated x and y coordinates of the original
    gps coordinates
    """
    path = "data/CEP/fixed_all.csv"
    df = cartesian_conversion(path)
    new_file_path = "../data/CEP/fixed_all_cartesian.csv"
    df.to_csv(new_file_path, index=False)
