import os
import json
import pandas as pd
from googleapiclient.discovery import build

def get_youtube(fbworkouts_path, youtube_csv_path):
    """
    Uses youtube links from fbworkouts.csv and writes to workouts_yt.csv
    with video title, published date, like/share/comment count info
    """
    # don't run if workouts_yt.csv already exist
    if os.path.isfile(youtube_csv_path):
        return

    # setup
    with open('config/api_key.json') as fh:
        api_key = json.load(fh)['api_key']
    service = build('youtube','v3', developerKey=api_key)
    fbworkouts = pd.read_csv(fbworkouts_path,encoding="ISO-8859-1")

    # intialize dct for output dataframe
    columns = ['title','published_at','view_count','like_count',
                'dislike_count','comment_count']
    dct = {}
    dct['workout_id'] = list(fbworkouts['workout_id'])
    for c in columns:
        dct[c] = []

    # properties to grab
    properties = 'snippet,statistics'

    # video ids
    yt_ids = fbworkouts['youtube_link'].str.replace(
        'https://www.youtube.com/watch?v=', '', regex=False).str.slice(0,11)

    # request 50 videos per request at a time (since max ids is 50)
    chunks = [yt_ids[x:x+50] for x in range(0, len(yt_ids), 50)]
    for chunk in chunks: # 6th and 8th element
        yt_ids = ','.join(list(chunk))

        # execute query
        response = service.videos().list(part=properties, id=yt_ids).execute()

        # populate dct
        for item in response['items']:
            snippet = item['snippet']
            stats = item['statistics']

            dct['title'].append(snippet['title'])
            dct['published_at'].append(snippet['publishedAt'])
            dct['view_count'].append(stats['viewCount'])
            dct['like_count'].append(stats['likeCount'])
            dct['dislike_count'].append(stats['dislikeCount'])
            dct['comment_count'].append(stats['commentCount'])

    # write data to csv
    pd.DataFrame(dct).to_csv(youtube_csv_path, index=False)
