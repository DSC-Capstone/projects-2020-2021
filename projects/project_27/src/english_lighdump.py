import requests
import urllib.request
import zipfile
from functools import reduce
import pandas as pd

def first_step(url_first, rorw, write_file):
  url = url_first 
  filehandle, _ = urllib.request.urlretrieve(url)
  zip_file_object = zipfile.ZipFile(filehandle, rorw)
  first_file = zip_file_object.namelist()[0]
  file1 = zip_file_object.open(first_file)
  content = file1.read()
  
  tempzip = open(write_file, "wb")
  tempzip.seek(0)
  tempzip.write(content)
  tempzip.seek(0)
   
  
  
def create_title_col(mal):
    mal.reset_index(drop=True,inplace=True)
    titles_df= mal[mal['revert'].isna()]
    titles=titles_df['date'].values
    
    start=titles_df.index[0]
    raw=np.array(titles_df.index)
    raw=np.append(raw,mal.index[-1])
    counts=np.diff(raw)-1
    counts[len(counts)-1]=counts[len(counts)-1]+1
    newmal=mal.iloc[start:,:]
    newmal=newmal.drop(index=titles_df.index)

    title_col=[]
    for i in range(len(titles)):
        each=[titles[i]]*counts[i]
        title_col.extend(each)

    newmal['title']=title_col
    return newmal
    
def merge_with_en(chunk):
    chunk.columns=['date','revert','edit','commentor']
    chunk['date'] = chunk['date'].apply(lambda x: str(x).replace('^^^_',''))
    chunk['date'] = chunk['date'].apply(lambda x: str(x).replace('T',' '))
    chunk['date'] = chunk['date'].apply(lambda x: str(x).replace('Z',''))
    newmal=create_title_col(chunk)
    otherinfo = pd.read_csv("test/output/otherinfo.csv")
    each=pd.merge(newmal,otherinfo,left_on=['date','commentor'],right_on=['date','commentor'],how='inner')
    return each
    
def concat_together(result,each):
    return pd.concat([result,each])
    


def english_ligh_dump(writelink, to_dataframe):
  chunk_list=pd.read_csv(writelink, sep=" ", header=None,chunksize=500000, error_bad_lines=False)
  processed_chunks = map(merge_with_en, chunk_list)
  result = reduce(concat_together, processed_chunks)
  result.reset_index(drop=True,inplace=True)
  thrid_result = result.to_csv(to_dataframe)
  return thrid_result
