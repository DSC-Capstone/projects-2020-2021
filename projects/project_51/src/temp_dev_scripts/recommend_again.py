from billboard import billboard
from sklearn import svm
import os
# import spotipy
# from spotipy.oauth2 import SpotifyOAuth
# from spotipy.oauth2 import SpotifyClientCredentials

#sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

# URI = 'http://localhost:8080'

# scope = " ".join(['playlist-modify-public',"user-top-read","user-read-recently-played","playlist-read-private"])

# username = 'skdud712'

# try:
#     token = spotipy.util.prompt_for_user_token(client_id = SPOTIPY_CLIENT_ID,
#                                                client_secret = SPOTIPY_CLIENT_SECRET,
#                                                redirect_uri = URI,
#                                                scope = scope,
#                                                username=username)
# except:
#     os.remove(f'cache-{username}')
#     token = spotipy.util.prompt_for_user_token(username=username)
    
# sp = spotipy.Spotify(auth=token)

playlists = sp.current_user_playlists()['items']
playlist_id = ''
for i in playlists:
# get latest playlist generated from the project; change this line to something else once we figure out how much listening history we would pull
    if i['name'] == 'capstone':
        playlist_id = i['id']
        
seed_artists = []
seed_tracks = []
#seed_genres = set()
for i in spotify.playlist_tracks(playlist_id=playlist_id)['items']:
    track = i['track']
    seed_artists += [track['artists'][0]['id']]
    seed_tracks.append(track['id'])
    #seed_genres.add(track['genre'])

bb = billboard()
f = bb.features
# liked songs
seed = f.loc[f['spotify_track_id'].isin(seed_tracks)]
# negative training set; fix to get songs in the same timeframe
random = f.loc[~f['spotify_track_id'].isin(seed_tracks) & ~f[f.columns[10:-1]].isnull()].sample(len(seed))

# train svm
X = np.concatenate((seed[seed.columns[10:-1]].values, random[random.columns[10:-1]].values))
y = [1 for i in range(30)] + [0 for i in range(30)]
clf = svm.SVC()
clf.fit(X, y)

# test set; fix to get songs in the same timeframe
test_rec = bb.getList(length=100)
test_f = f.loc[f['spotify_track_id'].isin(test_rec)]

new_tracks = []
for index, row in test_f.iterrows():
    test = row[10:-1].values
    if (~pd.isnull(test).any()):
        if clf.predict([test]) == 1:
            new_tracks.append(row['spotify_track_id'])

