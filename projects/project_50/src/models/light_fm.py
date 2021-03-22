import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from lightfm import LightFM


def light_fm(df):
    """
    Takes in data dictionary, and output lightfm's predictions for every user

    Note: this function is used for run.py (offline testing)
    """
    model = LightFM(loss='warp')

    model.fit(df['train_ui_matrix'])
    user_id = np.asarray(
        [u for u in range(df['user_item_interactions']['user_id'].nunique())])
    workout_id = np.asarray(
        [i for i in range(df['user_item_interactions']['workout_id'].nunique())])
    pred = [model.predict(int(i), workout_id) for i in user_id]

    return pred

def evaluate(df, pred, k=None):
    """
    Takes in data dictionary and returns average NDCG for LightFM
    :param df: data_dict
    :param pred: prediction from the function above
    :param k: any k value

    Note: this function is used for run.py (offline testing)
    """
    return ndcg_score(df['test_ui_matrix'].toarray(), pred, k)

def pred_i(df, user_id):
    """
    Takes in data dictionary and external user id, and outputs LightFM's predictions
    (converted to external workout ids) and their respective scores.

    Note: this function is deployed to web application
    """
    model = LightFM(loss='warp')

    model.fit(df['all_ui_matrix'])
    workout_ids = np.asarray(
        [i for i in range(df['user_item_interactions']['workout_id'].nunique())])

    # get LightFM scores, by internal indices
    scores = model.predict(get_internal_user_id(
        df['user_map'], user_id), workout_ids)

    # internal indices ordered by scores (descending)
    internal_indices_ranked = np.argsort(-scores)

    # LightFM scores corresponding to ranked indices
    scores_ranked = scores[internal_indices_ranked]

    # external indices order by scores (decending)
    external_indices_ranked = [get_external_workout_id(
        df['item_map'], i) for i in internal_indices_ranked]
    return external_indices_ranked, scores_ranked


def get_internal_workout_id(mapping, workout_id):
    """
    Helper fuction that takes in item mapping and external workout id, returns
    LightFM's corresponding internal indice for that workout
    """
    return mapping[workout_id]

def get_internal_user_id(mapping, user_id):
    """
    Helper fuction that takes in user mapping and external user id, returns
    LightFM's corresponding internal indice for that user
    """
    return mapping[user_id]

def get_external_workout_id(mapping, internal_workout_id):
    """
    Helper fuction that takes in item mapping and LightFM's internal indice
    for a workout, returns corresponding external indice for that workout
    """
    return {v: k for k, v in mapping.items()}[internal_workout_id]

def get_external_user_id(mapping, internal_user_id):
    """
    Helper fuction that takes in user mapping and LightFM's internal indice
    for a user, returns corresponding external indice for that user
    """
    return {v: k for k, v in mapping.items()}[internal_user_id]
