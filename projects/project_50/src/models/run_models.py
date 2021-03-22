import pandas as pd
import numpy as np
from top_popular import top_popular, evaluate_top_popular
from light_fm import light_fm, evaluate
import warnings

warnings.filterwarnings("ignore")


def run_models(data_dct, k=None):

    # Random #######################################################
    print("\nRunning Random...")

    user_id = np.asarray(
        [u for u in range(data_dct['user_item_interactions']['user_id'].nunique())])
    pred_random = np.array(
        [np.random.normal(0, 1, len(data_dct['item_map'].values())) for i in user_id])
    random_ndcg = evaluate(data_dct, pred_random, k)

    print('Average NDCG of Random: ' + str(random_ndcg))

    # Top Pop ######################################################
    print("\nRunning Top Popular...")

    top_pop_ndcg = evaluate_top_popular(data_dct['train_df'],
                                        data_dct['test_ui_matrix'],
                                        data_dct['item_map'], k)

    print('Average NDCG of Top Popular: ' + str(top_pop_ndcg))

    # LightFM ######################################################
    print("\nRunning LightFM...")

    pred_lightfm = light_fm(data_dct)
    light_fm_ndcg = evaluate(data_dct, pred_lightfm, k)

    print('Average NDCG of LightFM: ' + str(light_fm_ndcg))
