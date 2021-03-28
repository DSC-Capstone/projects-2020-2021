import pandas as pd
import numpy as np
import pickle
import os

def strip_special_chars(df, col):
    """
    Helper function: this will clean up a given column in a dataframe inplace,
    and will remove any character that is not alphanumerical, a comma, or a slash
    """
    df[col] = df[col].str.replace('[^a-zA-Z0-9/, ]', '', regex=True)

    # add space in body_focus column after special character is removed i.e UpperBody -> Upper Body
    if col == 'body_focus':
        df['body_focus'] = df['body_focus'].str.replace('B', ' B')
    return df

def clean_fbworkouts(fbworkouts_path, fbworkouts_clean_path):
    """
    Takes in fbworkouts.csv and outputs fbworkouts_clean.csv
    """
    # reads workouts_df
    workouts_df = pd.read_csv(fbworkouts_path, encoding="ISO-8859-1")

    # extracts the minutes from the column
    duration = workouts_df.duration.str.split().apply(lambda x: x[0] if x[1] == 'Minutes' else x)
    workouts_df.duration = duration.astype(int)

    def split(df, col):
        """
        Splits strings into list with pythonic naming (lowercase, slashes and
        spaces replaced with _) for a given column in a dataframe
        """
        df[col] = df[col].str.lower().replace('[ \/]', '_', regex=True)
        df[col+'_list'] = df[col].str.split(',_')
        df[col] = df[col].str.replace(',_',', ')
        return df

    # strip special characters and convert to list i.e. ['upper_body', 'total_body', 'lower_body', 'core'\
    for c in ['body_focus','training_type','equipment']:
        strip_special_chars(workouts_df, c)
        split(workouts_df, c)

    # converts the calories burned from a range to a numerical mean
    calories = workouts_df.calorie_burn.str.split('-')
    # calories_mean = calories.apply( lambda x: (float(x[0]) + float(x[1])) / 2 )
    # workouts_df.calorie_burn = calories_mean
    # workouts_df = workouts_df.rename(columns={"calorie_burn": "mean_calorie_burn"})

    workouts_loc = workouts_df.columns.get_loc("calorie_burn")
    workouts_df.insert(loc=workouts_loc + 1, column="min_calorie_burn", value=calories.apply( lambda x: + int(x[0]) ))
    workouts_df.insert(loc=workouts_loc + 2, column="max_calorie_burn", value=calories.apply( lambda x: + int(x[1]) ))
    workouts_df = workouts_df.drop(["calorie_burn"], axis=1)

    # OHE Encoder Function
    def OHEListEncoder(df, col, drop=True):
        """
        Given a dataframe and a column, return a OHE encoding of the column
        df: pandas dataframe
        col: str, name of the column to encode
        drop: Boolean, drops column from dataframe (default = True)
        """
        expanded_col = df[col].explode()
        if drop: df = df.drop([col], axis=1)
        return df.join(pd.crosstab(expanded_col.index, expanded_col))


    workouts_df = OHEListEncoder(workouts_df, 'body_focus_list')
    workouts_df = OHEListEncoder(workouts_df, 'training_type_list')
    # there is both a workout type and equipment named kettlebell, meaning that there will be overlap
    # therefore, we dropped the kettlebell from the "training_type", since you won't be doing
    # kettlebell exercises without the kettlebell; kettlebell will be encoded in the equipment section
    workouts_df = workouts_df.drop(['kettlebell'], axis=1, errors='ignore')
    workouts_df = OHEListEncoder(workouts_df, 'equipment_list')

    workouts_df = workouts_df.drop(['youtube_link'], axis=1)
    workouts_df.to_csv(fbworkouts_clean_path, index=False)

def create_metadata(fbworkouts_path, all_links_pickle_path, fbworkouts_meta_path, youtube_csv_path):
    """
    Takes in fbworkouts.csv and all_links.pickle and outputs fbworkouts_meta.csv
    """
    with open(all_links_pickle_path, 'rb') as file:
        links = pickle.load(file)

    # creates series of pickle links
    workout_fb_url = pd.Series(links)

    # loads in workout url and youtube url from fbworkouts.csv
    workouts_df = pd.read_csv(fbworkouts_path, encoding="ISO-8859-1")
    workout_ids = workouts_df.workout_id
    workout_yt_url = workouts_df.youtube_link

    # strip special characters i.e. 'Upper Body, Total Body'
    for c in ['body_focus','training_type','equipment']:
        strip_special_chars(workouts_df, c)

    # get cleaned body_focus, training_type, equipment columns
    workout_body_focus = workouts_df.body_focus
    workout_training_type = workouts_df.training_type
    workout_equipment = workouts_df.equipment

    # get workout name from fb_link
    youtube_df = pd.read_csv(youtube_csv_path)
    titles = youtube_df['title']

    # writes to pandas DataFrame
    meta_df_dict = {
        'workout_id': workout_ids,
        'workout_title': titles,
        'fb_link': workout_fb_url,
        'youtube_link': workout_yt_url,
        'body_focus': workout_body_focus,
        'training_type': workout_training_type,
        'equipment': workout_equipment
        }

    meta_df = pd.concat(meta_df_dict, axis=1)

    meta_df.to_csv(fbworkouts_meta_path, index=False)


def create_fbcommenters(comments_path, fbcommenters_path, d=0):
    """
    Takes in comments.csv and outputs fbcommenters.csv, which assigns id to each
    hash_id-profile combination.

    Note: only fbcommenters who commented at least d times are kept
    """
    comments_df = pd.read_csv(comments_path, usecols=['hash_id'])
    counts= comments_df.groupby('hash_id').size()
    more_than_d = counts[counts >= d].index

    dct = {
        'hash_id': more_than_d,
        'user_id': np.arange(1, len(more_than_d) + 1)
    }

    out = pd.DataFrame(dct)
    out.to_csv(fbcommenters_path, index=False)

def create_UI_interactions(comments_path, fbcommenters_path, user_item_interactions_path):
    """
    Outputs user_item_interactinos.csv, containing columns user_id and workout_id
    """
    comments_df = pd.read_csv(comments_path, usecols=['hash_id', 'workout_id']).drop_duplicates() # some users might comment twice on the same video
    fbcommenters_df = pd.read_csv(fbcommenters_path)
    merged_df = pd.merge(comments_df, fbcommenters_df, on="hash_id", how='inner')
    interactions_df = merged_df[['user_id','workout_id']].sort_values(['user_id','workout_id'])
    interactions_df.to_csv(user_item_interactions_path, index=False)

def fb_preprocessing(fbworkouts_path, fbworkouts_clean_path, comments_path, fbcommenters_path, user_item_interactions_path, fbworkouts_meta_path, all_links_pickle_path, youtube_csv_path, d=5):
    # create data/preprocessed folder if it doesn't yet exist
    dirname = os.path.dirname(fbworkouts_clean_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    clean_fbworkouts(fbworkouts_path, fbworkouts_clean_path)
    create_metadata(fbworkouts_path, all_links_pickle_path, fbworkouts_meta_path, youtube_csv_path)
    create_fbcommenters(comments_path, fbcommenters_path, d)
    create_UI_interactions(comments_path, fbcommenters_path, user_item_interactions_path)
