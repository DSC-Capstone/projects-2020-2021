import requests
import tarfile
import os

def sample_ibc(already_obtained="False"):
    '''
    Downloads sample_ibc dataset.
    For instructions on how to download the full dataset, see README.md
    '''
    if already_obtained == "False":
        print('NOTICE: Sample IBC data is being downloaded.')
        print('Please see README.md for instructions on how to obtain the full dataset.')
        print('Once downloaded, please set the config to reflect that change.')

        r = requests.get('https://people.cs.umass.edu/~miyyer/data/sample_ibc.tar.gz')
        if not os.path.exists('data/full_ibc'): # make the path if needed
            os.makedirs('data/full_ibc')
        with open('data/full_ibc/sample_ibc.tar.gz', 'wb') as f:
            f.write(r.content)

        tar = tarfile.open("data/full_ibc/sample_ibc.tar.gz", "r:gz") #extract
        tar.extractall(path='data/full_ibc/')
        os.rename('data/full_ibc/sampleData.pkl','data/full_ibc/ibcData.pkl')