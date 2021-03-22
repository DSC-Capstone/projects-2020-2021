from billboard import billboard
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from spotipy.oauth2 import SpotifyClientCredentials

SPOTIPY_CLIENT_ID='608e3b1bcba14f8bb1ed22fa44291d13'
SPOTIPY_CLIENT_SECRET='848e0d68cf6a4d00b07f583a69aa35dd'

#sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=SPOTIPY_CLIENT_ID, client_secret=SPOTIPY_CLIENT_SECRET))

URI = 'http://localhost:8080'

scope = " ".join(['playlist-modify-public',"user-top-read","user-read-recently-played","playlist-read-private"])

username = 'skdud712'

try:
    token = spotipy.util.prompt_for_user_token(client_id = SPOTIPY_CLIENT_ID,
                                               client_secret = SPOTIPY_CLIENT_SECRET,
                                               redirect_uri = URI,
                                               scope = scope,
                                               username=username)
except:
    os.remove(f'cache-{username}')
    token = spotipy.util.prompt_for_user_token(username=username)
    
spotify = spotipy.Spotify(auth=token)

x = billboard()
billboard_rec = x.getList(startY=2010, endY=2020, genre=['electronica'])

playlist = spotify.user_playlist_create(username, name='capstone', public=True, collaborative=False, description='')
spotify.user_playlist_add_tracks(user=username, 
                                 playlist_id=str(playlist['id']), # can't get this specific line to work for some reason f
                                 tracks=billboard_rec, 
                                 position=None)