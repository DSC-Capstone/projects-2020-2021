from app_parser import get_data
from hin_builder import get_features
from hindroid_etl import *

def run_etl(outfolder, parse_params, feature_params, hindroid_params=None):
    get_data(outfolder, **parse_params)
    get_features(outfolder, **feature_params)
    
    if hindroid_params is not None:
        build_from_folder(outfolder, redo=hindroid_params['redo'])
