import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os 
def Analysis(nonzero_link, zero_link):

  #first analysis for corr between M and sentiment score
  nzerom = pd.read_csv(nonzero_link)
  zerom = pd.read_csv(zero_link)
  nzero_title=nzerom.groupby('title').mean()
  X = nzero_title['M']
  Y = nzero_title['sentiment_score']
  plt.scatter(y=Y,x = X)
  plt.xlabel('M')
  plt.ylabel('sentiment_score')
  plt.title('M VS Sentiment')
  plt.savefig('test/output/M_VS_Sentiment.png')
  plt.close()

  
  #second analysis for example of Wooster Ohio
  zerom_withsen=zerom.loc[zerom['title']=='Wooster, Ohio']['sentiment_score']
  plt.hist(zerom_withsen)
  plt.title('Sentiment Score Distribution in One Article')
  plt.xlabel('Sentiment Score') 
  plt.ylabel('Comment Counts')

  
  plt.savefig('test/output/Wooster_example.png')
  plt.close()

  #third analysis for relationship between pageview and sentiment score
  all_views=nzerom.groupby('title').mean()
  all_views['views']=np.log(all_views['views'])

  y=all_views['views'].fillna(0)
  x=all_views['sentiment_score']
  plt.scatter(y=abs(y),x=abs(x))

  z = np.polyfit(x, y, 1)
  p = np.poly1d(z)
  plt.plot(x,p(x),"r--")

  plt.xlabel('Sum Sentiment Score')
  plt.ylabel('View Counts')
  plt.title('Sentiment VS View')
  plt.savefig('test/output/Sentiment vs View.png')
  plt.close()

def view_count_vs_m(nonzero_link, zero_link):
  nzerom = pd.read_csv(nonzero_link)
  zerom = pd.read_csv(zero_link)
  all_views=nzerom.groupby('title').mean()
  all_views['views']=np.log(all_views['views'])

  #fourth analysis for view counts with M
  logm = all_views.copy()
  logm['M'] = np.log(logm['M']+1)
  y_axis=logm['views'].fillna(0)
  x_axis=logm['M']
  plt.scatter(y=abs(y_axis),x=abs(x_axis))

  z = np.polyfit(x_axis, y_axis, 1)
  p = np.poly1d(z)
  plt.plot(x_axis,p(x_axis),"r--")

  plt.xlabel('M')
  plt.ylabel('View Counts')
  plt.title('View counts v.s. M')
  plt.savefig('test/output/Viewcounts v.s. M.png')
  plt.close()
  
  


