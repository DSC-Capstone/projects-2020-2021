# Brian Cheng
# Eric Liu
# Brent Min

# run.py is a file that 
# [TODO]

import argparse
import time

from src.data.run_data import run_data
from src.model.top_pop import top_pop
from src.model.cosine_rec import cosine_rec
from src.functions import get_params

def main(params=None):
    """
    The main function that runs all code. Typically, this function will be called from the command
    line so arguments will be read from there. Optionally, this function can be called from another
    file with input parameters to ignore command line arguments.

    :param:     params      Optional command line arguments in dictionary form. If not None, then
                            command line arguments will be ignored
    """
    # params will only be not None if this function is called from the website
    # in that case, change the behavior of the script accordingly
    if(params != None):
        if(params["recommender"] == "top_pop"):
            return top_pop(None, None, params)
        if(params['recommender'] == 'cosine_rec'):
            return cosine_rec(None, None, params)
        elif(params["recommender"] == "debug"):
            results = top_pop(None, None, params)
            results["notes"].append(f"\nInput is: {str(params)}")
            results["notes"].append(f"\nOutput is: {str(results['recommendations'])}")
            return results
        else:
            return str(params)

    # all command line arguments
    # for a description of the arguments, refer to the README.md or run "python run.py -h"
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", action="store_true", help="The program will run data " \
        "scraping code if this flag is present.")
    parser.add_argument("-c", "--clean", action="store_true", help="The program will run data " \
        "cleaning code if this flag is present.")
    parser.add_argument("--data-config", default=["config/data_params.json"], type=str, nargs=1,
        help="Where to find data parameters. By default \"config/data_params.json\".")
    parser.add_argument("--web-config", default=["config/web_params.json"], type=str, nargs=1,
        help="Where to find simulated web parameters. By default \"config/web_params.json\".")
    parser.add_argument("--top-pop", action="store_true", help="The program will print the " \
        "the top 10 most popular/well received climbs based on the web params.")
    parser.add_argument("--cosine", action="store_true", help="The program will print the " \
        "the top 10 most similar climbs based on the web params.")
    parser.add_argument("--test", action="store_true", help="The program will run all code in a " \
        "simplified manner. If this flag is present, it will run top popular and cosine "
        "recommenders using pre-cleaned data in MongoDB. Using the --test flag is equivalent to " \
        "running the project using \"python run.py --top-pop --cosine --debug\".")
    parser.add_argument("--delete", action="store_true", help="The program will wipe out all " \
        "data from MongoDB. This will not work since the MongoDB login is read only.")
    parser.add_argument("--upload", action="store_true", help="The program will upload cleaned " \
        "data to MongoDB. This will not work since the MongoDB login is read only.")
    parser.add_argument("--debug", action="store_true", help="The program will do various print " \
        "statements if this flag is present.")

    # parse all arguments
    args = vars(parser.parse_args())

    # override args if the test flag is present
    # for a more complete description of what this is doing, refer to README.md or look at 
    # https://github.com/DSC180-RC/Rock-Climbing-Recommender/blob/master/README.md
    if(args["test"]):
        # override command line args
        args["data"] = False
        args["clean"] = False
        args["data_config"] = ["config/data_params.json"]
        args["web_config"] = ["config/web_params.json"]
        args["top_pop"] = True
        args["cosine"] = True
        args["delete"] = False
        args["upload"] = False
        args["debug"] = True

    # debug print after all command line args are set
    if(args["debug"]):
        print("Command line args:")
        print(args)
        print()
    
    # read the config files
    data_params = get_params(args["data_config"][0])
    web_params = get_params(args["web_config"][0])

    # debug print for config files
    if(args["debug"]):
        print("Data config file:")
        print(data_params)
        print()
        print("Web config file:")
        print(web_params)
        print()

    # run data code
    run_data(data_params, args)

    # run top pop code if requested
    if(args["top_pop"]):
        print("Top pop results:")
        print(top_pop(args, data_params, web_params))
        print()

    # run top pop code if requested
    if(args["cosine"]):
        print("Cosine rec results:")
        print(cosine_rec(args, data_params, web_params))
        print()

# run.py cannot be imported as a module
if __name__ == '__main__':
    # keep track of how long the program has run for
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
