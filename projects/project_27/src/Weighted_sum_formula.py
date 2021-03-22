import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def weighted_sum(revert,views,sentiment,M):
    return views*(revert+sentiment)+M


def weighted_sum_formula(nzerom_link,zerom_link):
  nzerom=pd.read_csv(nzerom_link)
  zerom=pd.read_csv(zerom_link)
  nzerom['M']=np.log(nzerom['M'])
  zerom['M']=np.log(zerom['M']+1)
  
  nzerom['views']=np.log(nzerom['views'])
  zerom['views']=np.log(zerom['views'])
  
  nzerom['sentiment_score']=abs(nzerom['sentiment_score'])
  zerom['sentiment_score']=abs(zerom['sentiment_score'])
  
  nzero_title=nzerom.groupby('title').agg({'revert':'mean', 
                         'views':'mean', 
                         'sentiment_score':'mean', 
                         'M': 'mean'})
  zero_title=zerom.groupby('title').agg({'revert':'mean', 
                         'views':'mean', 
                         'sentiment_score':'mean', 
                         'M': 'mean'})
  
  with_weighted=zero_title.apply(lambda x: weighted_sum(x['revert'],x['views'],x['sentiment_score'],x['M']), axis=1)
  zero_title['Weighted_sum']=with_weighted
  
  low_m=zero_title.sort_values('M',ascending=True).head(100)
  threshold=zero_title['Weighted_sum'].quantile(0.95)
  opposite=low_m.loc[low_m['Weighted_sum']>=threshold]
  opposite.sort_values('Weighted_sum',ascending=False)
  
  prev=zero_title['M'].sort_values(ascending=False)
  after=zero_title['Weighted_sum'].sort_values(ascending=False)
  Ax=range(zero_title.shape[0])
  plt.plot(Ax,prev,label='M',c='orange')
  plt.legend(loc="upper right")
  plt.ylabel('M Statistic')
  plt.title('M score performance')
  plt.savefig('test/output/M score performance.png')
  plt.close()
  
  prev=zero_title['M'].sort_values(ascending=False)
  after=zero_title['Weighted_sum'].sort_values(ascending=False)
  Ax=range(zero_title.shape[0])
  plt.plot(Ax,after,label='Weighted Sum',c='blue')
  plt.legend(loc="upper right")
  plt.ylabel('Weighted Sum Score')
  plt.title('Weighted sum formula performance')
  plt.savefig('test/output/Weighted sum formula performance.png')
