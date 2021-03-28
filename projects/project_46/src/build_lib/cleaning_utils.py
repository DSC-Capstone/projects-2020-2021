import pandas as pd

#---------------------------------------------------- CLEANING BILLBOARD DATA ----------------------------------------------------

def clean_billboard(billboard_songs, billboard_features):
    
    billboard_features = billboard_features.dropna(subset=['spotify_track_id', 'spotify_genre']).drop_duplicates(subset='spotify_track_id')
    billboard_features['spotify_genre'] = [x.strip('[]').strip('\'').split('\', \'') for x in billboard_features['spotify_genre']]

    return billboard_songs, billboard_features


#---------------------------------------------------- CLEANING LAST.FM DATA ----------------------------------------------------

def clean_lastfm(user_profile_df, user_artist_df):
    
    # Remove observations with null values
    user_profile_df = user_profile_df[['user_id', 'age', 'country']].dropna().reset_index(drop=True)
    # Select rows with users from US-recommendation task targeted to US users
    user_profile_df = user_profile_df[user_profile_df['country'] == 'United States']
    cleaned_user_profile_df = user_profile_df[user_profile_df['age'] > 0]
    

    # Drop rows with missing values
    user_artist_df = user_artist_df[['user_id', 'artist_id', 'artist_name', 'plays']].dropna().reset_index(drop=True)
    # Extract listening histories from US users 
    cleaned_user_artist_df = extract_histories(user_artist_df, cleaned_user_profile_df)

    return cleaned_user_profile_df, cleaned_user_artist_df

# Given a set of users, pull their listening histories
def extract_histories(user_artist_df, user_profile_df):
    # Extract listening histories from users selected
    extracted_history = user_artist_df[user_artist_df['user_id'].isin(user_profile_df['user_id'])]
    return extracted_history