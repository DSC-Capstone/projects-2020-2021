import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/utils')
sys.path.insert(0, 'src/models')

from clean import remove_data
from scrape import scrape_data
from fbpreprocessing import fb_preprocessing
from model_preprocessing import get_data
from run_models import run_models
from youtube import get_youtube

def gather_data(data_params):
    with open('config/chromedriver.json') as fh:
        chromedriver_path = json.load(fh)['chromedriver_path']

    print("Scraping data...")
    scrape_data(chromedriver_path,
                data_params['all_links_pickle_path'],
                data_params['fbworkouts_path'],
                data_params['comments_path'])
    print("Scraping done.")

    print("Querying Youtube API...")
    get_youtube(data_params['fbworkouts_path'],
                data_params['youtube_csv_path'])
    print("Querying done")

def preprocess(data_params, d):
    print("Preprocessing...")
    fb_preprocessing(
        fbworkouts_path = data_params['fbworkouts_path'],
        fbworkouts_clean_path = data_params['fbworkouts_clean_path'],
        comments_path = data_params['comments_path'],
        fbcommenters_path = data_params['fbcommenters'],
        user_item_interactions_path = data_params['user_item_interactions_path'],
        fbworkouts_meta_path = data_params['fbworkouts_meta_path'],
        all_links_pickle_path = data_params['all_links_pickle_path'],
        youtube_csv_path = data_params['youtube_csv_path'],
        d=d
        )
    print("Data preprocessing done.")

def run_model(data_params, k):
    data = get_data(data_params['user_item_interactions_path'])
    run_models(data, k=k)

def main(targets):
    if 'clean' in targets:
        remove_data()
        print("Data cleaned.")
        return

    if 'test' in targets:
        with open('config/test-params.json') as fh:
            data_params = json.load(fh)

        preprocess(data_params, d=0)
        run_model(data_params, k=None)
        return

    with open('config/data-params.json') as fh:
        data_params = json.load(fh)

    if 'all' in targets:
        gather_data(data_params)
        preprocess(data_params, d=5)
        run_model(data_params, k=20)
        return

    if 'data' in targets:
        gather_data(data_params)
        preprocess(data_params, d=5)

    if 'model' in targets:
        run_model(data_params, k=20)

    return


if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
