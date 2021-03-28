import os
import requests
import gzip
import pandas as pd

def get_data(file, out_path):
    print('Downloading Data')
    if file == 'DBLP.txt':
        url = 'http://dmserv2.cs.illinois.edu/data/DBLP.txt.gz'
        r = requests.get(url, allow_redirects=True)
        open('DBLP.txt.gz', 'wb').write(r.content)
        sentences = []
        f = gzip.open('DBLP.txt.gz', 'rb')
        for line in f.readlines():
            sentences.append(str(line))
        f.close()
        data = pd.DataFrame()
        data['sentence'] = sentences
        data['sentence'] = data['sentence'].apply(lambda x:x[2:-1])
        data['sentence'] = data['sentence'].apply(lambda x:x.strip('\\n'))
        data['sentence'] = data['sentence'].apply(lambda x: x.lower())
        data['sentence'] = data['sentence'].apply(lambda x: x.replace('(',''))
        data['sentence'] = data['sentence'].apply(lambda x: x.replace(')',''))
        data['sentence'] = data['sentence'].apply(lambda x: x.replace('?',''))
        data['sentence'] = data['sentence'].apply(lambda x: x.replace('@',''))
        data['sentence'] = data['sentence'].apply(lambda x: x.replace(':',''))
        data['sentence'] = data['sentence'].apply(lambda x: x.replace(',',''))
        data = data[data['sentence']!='']
        data = data.reset_index(drop = True)
        data.to_csv(out_path + file,index=None, header = None, sep='\t')
        os.remove('DBLP.txt.gz')
        print('Done')

    if file == "DBLP.5K.txt":
        data_kk = pd.read_csv("AutoPhrase/data/EN/DBLP.5K.txt", header = None, names=['sentence'])
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x:x.strip('\\n'))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.lower())
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace('(',''))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace(')',''))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace('?',''))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace('@',''))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace(':',''))
        data_kk['sentence'] = data_kk['sentence'].apply(lambda x: x.replace(',',''))
        data_kk = data_kk[data_kk['sentence']!='']
        data_kk = data_kk.reset_index(drop = True)
        data_kk.to_csv(out_path + file, index=None, header = None, sep='\t')
        print('Done')

    return 
