# Brian Cheng
# Eric Liu
# Brent Min

# run_data.py collates all data scraping and cleaning code into one convenient place

from src.data.get_raw_data import get_raw_data
from src.data.get_clean_data import get_clean_data
from src.data.delete_data import delete_data
from src.data.upload_data import upload_data
from src.functions import *

from pymongo import MongoClient

def run_data(data_params, args):
    """
    This function collates all scraping and cleaning code into a single function called by run.py

    :param:     data_params     The data parameters from the config file
    :param:     args            The command line input parameters. This tells us whether or not to
                                run data scraping/cleaning code
    """
    #the mongo connection
    client_url = 'mongodb+srv://DSC102:Kw4ngOgiLl6mPi5r@cluster0.4gstr.mongodb.net/MountainProject?retryWrites=true&w=majority'
    my_client = MongoClient(client_url)

    # first check that we want to run some data scraping/cleaning code
    if(args["data"] or args["clean"] or args["delete"] or args["upload"]):
        # TODO: Delete the data folders in order to empty them, since if the user requests that data
        #       scraping code be run, then overwrite existing data

        # create the folders in which to save data if the folders do not exist
        check_folder(data_params["raw_data_folder"])
        check_folder(data_params["clean_data_folder"])
        
        # get raw data if requested from the command line
        if(args["data"]):
            get_raw_data(data_params)

        # process raw data into cleaned data if requested from the command line
        if(args["clean"]):
            get_clean_data(data_params)

        if(args["delete"]):
            delete_data(data_params, my_client)

        if(args["upload"]):
            upload_data(data_params, my_client)
    