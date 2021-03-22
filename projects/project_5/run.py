import sys
import os
sys.path.insert(0, 'src')
import env_setup
import etl_election
import etl_news
import eda
import similarity
import hashtags
import pandas as pd
from utils import get_project_root, load_config
import json



def main(targets):
    """Runs the main project pipeline project, given targets."""
    root = get_project_root()

    if 'setup-dsmlp' in targets:
        env = load_config("config/env_dsmlp.json")
        env_setup.setup_dsmlp(**env)

    if 'test' not in targets:
        env_setup.auth()
        env_setup.make_datadir()

    if 'all' in targets:
        targets = ['data', 'eda','compile', 'embed']

    if 'data' in targets:
        config = load_config('config/news_params.json')
        etl_election.generate_dataset(trange='all')
        etl_election.rehydrate_tweets()
        etl_news.get_news_data(**config)
        

    if 'eda' in targets:
        eda.main()
    
    if 'compile' in targets:
        config = json.load(open('config/vector_compile.json'))
        kws = config['keywords']
        save_path = config['save_path']
        vector_fp = config['vector_fp']
        user_fp = config['user_fp']
        fig_fp = config['fig_path']
        ht_vector = hashtags.count_features(mode = 'hashtag', normalize=False, case_sensitive=False, top_k=None)
        ht_vector.to_json(vector_fp)
    
    if 'embed' in targets:
        config = json.load(open('config/vector_compile.json'))
        kws = config['keywords']
        save_path = config['save_path']
        vector_fp = config['vector_fp']
        # this line (user_fp) may need change for pointers towards locations of 
        users_fp = config['user_fp']
        fig_path = config['fig_path']
        vector = pd.read_json('vector.json', typ = 'series')
        
        # grid search chart generation
        normalize_list = [True,False]
        top_k_list = [300,None]
        kws_list = [kws,None]
        for normalize in normalize_list:
            for top_k in top_k_list:
                for kws in kws_list:
                    similarity.plot_embedding(users_fp, vector, save_path, fig_path=fig_path, normalize=normalize, kws=kws, top_k=top_k)
        

    if 'test' in targets:
        config = load_config('config/test_params.json')
        etl_news.get_news_data(**config)

        eda.main(test=True)

        config = load_config('config/test_params.json')
        etl_news.get_news_data(**config)
        test_vector = hashtags.count_features([os.path.join(root, 'test', 'testdata', 'test_election.jsonl')])
        result_path = os.path.join(root, 'test', 'testreport')
        test_vector.to_json(os.path.join(result_path, 'election_vector.json'))
        fig_path = os.path.join(root, 'test', 'testreport')
        save_path = os.path.join(fig_path, 'news_vector.json')
        users_fp = os.path.join(root, 'test', 'testdata')
        similarity.plot_embedding(users_fp, test_vector, save_path, fig_path=fig_path, normalize=False)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
