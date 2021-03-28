import sys
import json
from src.etl import get_data, download_dataset
from src.eda import generate_figures
from src.classification import model
from src.clustering import run_clustering


def main(targets):
    if "test" in targets:
        targets = ["data", "eda", "classification", "clustering"]
        etl_params = json.load(open("config/etl_test.json"))
        eda_params = json.load(open("config/eda_test.json"))
    else:
        etl_params = json.load(open("config/etl.json"))
        eda_params = json.load(open("config/eda.json"))

    autophrase_params = json.load(open("config/autophrase.json"))
    clf_params = json.load(open("config/classification.json"))
    clustering_params = json.load(open("config/clustering.json"))

    if "all" in targets:
        targets = ["data", "eda", "classification", "clustering"]

    if "download" in targets:
        download_dataset()

    if "data" in targets:
        get_data(autophrase_params, **etl_params)

    if "eda" in targets:
        generate_figures(**eda_params)

    if "classification" in targets:
        model(clf_params)

    if "clustering" in targets:
        run_clustering(clustering_params)


if __name__ == "__main__":
    main(sys.argv[1:])
