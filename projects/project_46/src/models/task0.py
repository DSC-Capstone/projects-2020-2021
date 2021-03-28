import pandas as pd
import numpy as np
import datetime

class billboard:
    def __init__(self, billboard_songs, billboard_features):
        
        # Column must be converted to datetime any time the file is read in
        billboard_songs['WeekID'] = pd.to_datetime(billboard_songs.reset_index()['WeekID'])

        self.billboard_songs = billboard_songs
        self.billboard_features = billboard_features
        
    def weeklyAvg(self):
        # average weekly position
        avg_pos = self.billboard_songs[['WeekID', 'Week Position', 'SongID']].groupby(by=['SongID']).mean()
        # first week the track appeared in the chart
        minweek = self.billboard_songs[['WeekID', 'SongID']].groupby(by=['SongID']).min().rename(columns={'WeekID':'firstWeekID'})
        # last week the track appeared in the chart
        maxweek = self.billboard_songs[['WeekID', 'SongID']].groupby(by=['SongID']).max().rename(columns={'WeekID':'lastWeekID'})
        # total # of weeks the track was in the chart
        max_occ = self.billboard_songs[['SongID','Instance','Weeks on Chart']].groupby(by=['SongID']).max()

        stats = avg_pos.join(minweek).join(maxweek).join(max_occ)
        self.data = self.billboard_features.join(stats, on='SongID').rename(columns={'Week Position':'Avg Weekly'})

    def getList(self, length=30, age=None, genre=[], artist=[]):
        
        # As a default just use songs from the current year
        startY = 2019
        endY = datetime.datetime.today().year

        AGE_LOWER_BOUND = 15
        AGE_UPPER_BOUND = 30
        if age:
            # Determining time range for song recommendations
            current_year = datetime.datetime.today().year
            startY = current_year - abs(age - AGE_LOWER_BOUND)
            endY = current_year - abs(age - AGE_UPPER_BOUND)
        
        # songs should have left chart after lower bound (e.g. 2019 songs should still be on chart after 2019/1/1)
        lowerBound = datetime.datetime(startY, 1, 1)
        # songs should have entered chart before upper bound (e.g. 2019 songs should have been on chart before 2019/12/31)
        upperBound = datetime.datetime(endY, 12, 31)

        self.weeklyAvg()

        data = self.data
        filter_t = data[(data['firstWeekID'] < upperBound) & (data['lastWeekID'] > lowerBound)]
        
        filter_g = filter_t[filter_t.spotify_genre.apply(lambda x: bool(set(x) & set(genre)))]
        filter_a = filter_t[filter_t.Performer.apply(lambda x: bool(set(x) & set(artist)))]
        
        playlist = filter_g.append(filter_a)
        
        if len(playlist) < length:
            playlist = filter_t
        
        playlist.sort_values(['Instance','Avg Weekly','Weeks on Chart'], ascending=[True,True,False], inplace=True, ignore_index=True)
        
        return playlist['spotify_track_id'][:length].to_list()