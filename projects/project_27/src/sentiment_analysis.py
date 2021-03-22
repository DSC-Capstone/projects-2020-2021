from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np

def sentiment_analysis(inputlink,towardlink):
  last_dataf = pd.read_csv(inputlink)
  new_dictionary = {}
  for i in np.arange(last_dataf.shape[0]):
      comments = last_dataf.iloc[i]['comment']
      analyzer = SentimentIntensityAnalyzer()
      try:
          vader_score = analyzer.polarity_scores(comments)
      except:
          new_dictionary[comments]=0
      new_dictionary[comments] = vader_score['compound']
  
  null_dataframe = pd.DataFrame()
  null_dataframe['comment'] = new_dictionary.keys()
  null_dataframe['sentiment_score'] = new_dictionary.values()
  
  score_dataf = pd.merge(last_dataf,null_dataframe,on = 'comment', how = 'left')
  result = score_dataf.to_csv(towardlink, index = False)
  
  return result
