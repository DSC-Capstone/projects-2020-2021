import sys
import json
import os
import pandas as pd
import requests
import spotipy
from collections import defaultdict
import scipy.sparse as sparse
import numpy as np
import random
import implicit
from sklearn.preprocessing import MinMaxScaler
import ipywidgets
from ipywidgets import FloatProgress

def extract_users(df, age, age_range):
    
    # Build age range for users similar to parents
    start = age - age_range
    end = age + age_range

    # Select users from parents' age range
    users_selected = df[(df['age'] >= start) & (df['age'] <= end)].reset_index(drop=True)    
    return users_selected

def extract_histories(df, users):
    
    # Extract listening histories from users selected
    extracted_history = df[df['user_id'].isin(users['user_id'])]
    return extracted_history


def prepare_dataset(extracted_history):
    ap = extracted_history
    playCount = ap.plays
    
    #Normalize play count through min-max scaling
    normalizedCount = (playCount - playCount.min()) / (playCount.max() - playCount.min())
    ap = ap.assign(playCountScaled=normalizedCount)

    ap = ap.drop_duplicates()
    grouped_df = ap.groupby(['user_id', 'artist_id', 'artist_name']).sum().reset_index()

    # Assign categories to each user and artist-data preparation for implicit algorithm
    grouped_df['artist_name'] = grouped_df['artist_name'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].astype("category")
    grouped_df['artist_id'] = grouped_df['artist_id'].astype("category")
    grouped_df['user_id'] = grouped_df['user_id'].cat.codes
    grouped_df['artist_id'] = grouped_df['artist_id'].cat.codes
    return grouped_df

def parse_playlist_ids(response):
    # Pull playlist info
    playlist_ids = []
    for item in response['items']:
        pid = item['id']

        playlist_ids.append(pid)
    return playlist_ids

def parse_track_info(response):
    # Pull track, artist, and album info for each track
    track_names = []
    artist_names = []
    album_names = []
    for item in response['items']: 
        
        if item is None or item['track'] is None:
            continue
        # Gets the name of the track
        track = item['track']['name']
        # Gets the name of the album
        album = item['track']['album']['name']
        # Gets the name of the first artist listed under album artists
        artist = item['track']['album']['artists'][0]['name']
            
        track_names.append(track)
        album_names.append(album)
        artist_names.append(artist) 
    return track_names, album_names, artist_names

def pull_user_playlist_info(sp, user_artist_df):
    r = sp.current_user_playlists()
    
    # Pull user Spotipy playlists
    playlist_ids = parse_playlist_ids(r)


    # Pull all the tracks from a playlist
    tracks = []
    albums = []
    artists = []

    # Loop through each playlist one by one
    for pid in playlist_ids:
        # Request all track information
        r = sp.playlist_items(pid)
            
        tracks_pulled, albums_pulled, artists_pulled = parse_track_info(r)
        artists.extend(artists_pulled)
    
    
    # Artists the user has listened-normalized frequency
    playlist_artists = pd.Series(artists)
    playlist_grouped = playlist_artists.value_counts(normalize=True)
    
    # Find current user and add entries for each listened artist
    no_artist = playlist_grouped.shape[0]
    curr_user = user_artist_df.iloc[-1]['user_id'] + 1
    curr_user_id = [curr_user] * no_artist

    playlist_df = pd.DataFrame(playlist_grouped, columns=['playCountScaled']) 
    playlist_df.reset_index(level=0, inplace=True)
    playlist_df.columns = ['artist_name', 'playCountScaled']
    playlist_df['user_id'] = pd.Series(curr_user_id)

    # Alter ordering of the column-left most is user id
    cols = playlist_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    playlist_df = playlist_df[cols]

    playlist_df['artist_name'] = playlist_df['artist_name'].str.lower()
    
    # Match artist to artist id in the initial user-artist dataframe
    artist_pairing = dict(zip(user_artist_df.artist_name, user_artist_df.artist_id))
    playlist_df['artist_id'] = playlist_df['artist_name'].map(artist_pairing)
    playlist_df = playlist_df.dropna().reset_index(drop=True)
    playlist_df['artist_id'] = playlist_df['artist_id'].astype(int)
    return playlist_df, curr_user

def updated_df_with_user(user_artist_df, playlist_df):
    
    # Add user playlist listening history to the LastFM data
    # User requesting playlist is last user_id-added 1 user
    updated_df = user_artist_df.append(playlist_df)
    updated_df['artist_name'] = updated_df['artist_name'].astype("category")
    updated_df['user_id'] = updated_df['user_id'].astype("category")
    updated_df['artist_id'] = updated_df['artist_id'].astype("category")
    updated_df['user_id'] = updated_df['user_id'].cat.codes
    updated_df['artist_id'] = updated_df['artist_id'].cat.codes
    return updated_df

def build_implicit_model(user_artist_df, alpha):
    
    # Build user-item and item-user interaction matrices
    sparse_artist_user = sparse.csr_matrix((user_artist_df['playCountScaled'].astype(float), (user_artist_df['artist_id'], user_artist_df['user_id'])))
    sparse_user_artist = sparse.csr_matrix((user_artist_df['playCountScaled'].astype(float), (user_artist_df['user_id'], user_artist_df['artist_id'])))
    # Use implicit model with alternating least squares
    model = implicit.als.AlternatingLeastSquares(factors=20, regularization=0.1, iterations=50)
    data = (sparse_artist_user * alpha).astype('double')
    # Fit the model
    model.fit(data)
    
    user_vecs = model.user_factors
    artist_vecs = model.item_factors
    return sparse_user_artist, sparse_artist_user, user_vecs, artist_vecs

def get_related_artists(sp, uri):
    related = sp.artist_related_artists(uri)
    related_lst = []
    for artist in related['artists'][:6]:
        related_lst.append(artist['name'])
    return related_lst

def get_top_tracks(sp, uri):
    top_tracks = sp.artist_top_tracks(uri)
    top_lst = []
    for track in top_tracks['tracks'][:5]:
        top_lst.append(track['name'])
    return top_lst

def recommend(sp, user_id, sparse_user_artist, user_vecs, artist_vecs, user_artist_df, num_contents=100):
    #Use user-item interactions
    user_interactions = sparse_user_artist[user_id,:].toarray()
    user_interactions = user_interactions.reshape(-1) + 1
    user_interactions[user_interactions > 1] = 0
    rec_vector = user_vecs[user_id,:].dot(artist_vecs.T)
    min_max = MinMaxScaler()
    rec_vector_scaled = min_max.fit_transform(rec_vector.reshape(-1,1))[:,0]
    recommend_vector = user_interactions * rec_vector_scaled
    content_idx = np.argsort(recommend_vector)[::-1][:num_contents]
    # Build dataframe of recommended artist features-genres, top tracks, etc
    artists = []
    artist_uris = []
    artist_genres = []
    artist_related_artists = []
    artist_related_uris = []
    artist_top_tracks = []
    scores = []
    for idx in content_idx:
        artist = user_artist_df.artist_name.loc[user_artist_df.artist_id == idx].iloc[0]
        try:
            artist_uri = sp.search(artist, type='artist')['artists']['items'][0]['uri']
        except:
            continue
        artist_info = sp.artist(artist_uri)
        artist_genre = artist_info['genres']
        artist_related = get_related_artists(sp, artist_uri)
        artist_tracks = get_top_tracks(sp, artist_uri)
        artists.append(user_artist_df.artist_name.loc[user_artist_df.artist_id == idx].iloc[0])
        artist_uris.append(artist_uri)
        artist_genres.append(artist_genre)
        artist_related_artists.append(artist_related)
        related_uris = []
        for artist in artist_related:
            try:
                request = sp.search(artist, type='artist')
                related_uri = request['artists']['items'][0]['uri']
                related_uris.append(related_uri)
            except:
                continue
        artist_related_uris.append(related_uris)
        artist_top_tracks.append(artist_tracks)
        scores.append(recommend_vector[idx])
    
    #Outputted recommended artists along with genre info and scores
    recommendations = pd.DataFrame({'artist_name': artists, 'artist_uri': artist_uris, 'artist_genres': artist_genres, 'artist_top_tracks': artist_top_tracks, 'artists_related': artist_related_artists, 'artists_related_uris': artist_related_uris, 'score': scores})
    return recommendations

def get_top_recommended_tracks(sp, recommendations, genre_selection, N):
    # Filter recommended artists by genre
    filtered_recommendations = recommendations[recommendations.artist_genres.apply(lambda x: bool(set(x) & set(genre_selection)))]
    top_recommended_tracks = pd.DataFrame(filtered_recommendations['artist_top_tracks'].explode())
    
    #Find related artists to generate more songs-add those to filtered recommended songs
    if len(filtered_recommendations) < 10:
        related_artist_names = list(filtered_recommendations['artists_related'].explode())
        related_artists_lst = list(filtered_recommendations['artists_related_uris'].explode())
        related_tracks_lst = []
        for artist in related_artists_lst:
            artist_top_tracks = get_top_tracks(sp, artist)
            related_tracks_lst.extend(artist_top_tracks)
    
    recommended_lst = top_recommended_tracks['artist_top_tracks'].tolist()

    if len(top_recommended_tracks) < N:
        recommended_lst.extend(related_tracks_lst)
     
    top_recommended_df = pd.DataFrame(recommended_lst, columns=['song_recommendations'])
    # Get the top N tracks
    tracks_output = top_recommended_df.reset_index(drop=True)[:N]
    return tracks_output
    
    
    



