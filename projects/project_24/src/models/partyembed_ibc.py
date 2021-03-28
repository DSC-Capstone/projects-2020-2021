import sys

sys.path.insert(1, 'full_ibc')
sys.path.insert(1, 'partyembed')
sys.path.insert(1, 'data/full_ibc')

import tarfile
import numpy as np
import _pickle as cPickle
import subprocess
subprocess.call('git clone https://github.com/lrheault/partyembed.git', shell = True)
from partyembed.explore import Explore
import pandas as pd
import string
import shutil
import os
import requests


def interpret_ibc(temp_directory = 'data/temp/', out_directory = 'data/out/', agg_func = 'mean',ibc_path='data/full_ibc/ibcData.pkl', test=False):
    '''
    This function will write a series of TSVs interpreting each item in the IBC
    '''
    # https://stackoverflow.com/questions/31813504/move-and-replace-if-same-file-name-already-exists
    # IBC code was written for an older version of python, replace it with updated files
    shutil.copyfile(os.path.join('src/models', 'loadIBC.py'), os.path.join('data/full_ibc', 'loadIBC.py'))
    shutil.copyfile(os.path.join('src/models', 'treeUtil.py'), os.path.join('data/full_ibc', 'treeUtil.py'))
    shutil.copyfile(os.path.join('src/models', 'loadIBC.py'), os.path.join('loadIBC.py'))
    shutil.copyfile(os.path.join('src/models', 'treeUtil.py'), os.path.join('treeUtil.py'))
    
    import data.full_ibc.loadIBC
    import data.full_ibc.treeUtil
    import loadIBC
    import treeUtil
    
    if not os.path.exists(temp_directory):
        os.makedirs(temp_directory)
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    
    # https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
    # see remainder of the code at the bottom of this file
    file_id = '17DWO_2UyjEZ9HnH3AACSdH3JZLpvb4bP'
    destination = 'partyembed_models.tar.gz'
    download_file_from_google_drive(file_id, destination)
    tar = tarfile.open("partyembed_models.tar.gz", "r:gz") #extract
    tar.extractall(path='partyembed/partyembed/models')
    for i in os.listdir('partyembed/partyembed/models/partyembed_models'):
        shutil.move(os.path.join('partyembed/partyembed/models/partyembed_models',i), os.path.join('partyembed/partyembed/models/',i))
    
    [lib, con, neutral] = cPickle.load(open(ibc_path, 'rb')) # load IBC data
    m = Explore(model = 'House') # set up partyembed model
    word_dict = {} #make the dictionary
    
    if not test:
        for pref, clist in zip(['lib','con','neutral'],[lib,con,neutral]):
            filenumbers = generate_filenumbers(np.arange(100,100*(len(clist) // 100)+100,100)) #generate indices for upcoming loops
            filename_suffixes = [i[1] for i in filenumbers[:-1]] + [100*(len(clist) // 100)] #generate suffixes for filenames
            filenames = []
            for i in filename_suffixes:
                filenames.append(pref + '-' + str(i) + '.tsv') #create the full filenames
            for i,j in zip(filenames, filenumbers):
                chunk_processor(i, clist, j[0], j[1], word_dict, m, temp_directory)
    
        #combine all of the files into x and y variables
        liby,libx = combiner('lib',100*(len(lib) // 100) + 100)
        cony,conx = combiner('con',100*(len(con) // 100) + 100)
        neuy,neux = combiner('neutral',100*(len(neutral) // 100) + 100)
        x = libx+conx+neux
        y = liby+cony+neuy

        true_x = [func(i.split(', '),agg_func) for i in x]
    
    if test:
        for clist in [lib,con,neutral]:
            for i,j,k in zip(['lib-100.tsv','con-100.tsv','neu-100.tsv'], [[None,None],[None,None],[None,None]],[lib,con,neutral]):
                word_dict = chunk_processor(i, k, j[0], j[1], word_dict, m, temp_directory)
        
        #combine all of the files into x and y variables
        liby,libx = test_combiner('lib',100)
        cony,conx = test_combiner('con',100)
        neuy,neux = test_combiner('neu',100)
        x = libx+conx+neux
        y = liby+cony+neuy

        true_x = [func(i.split(', '),agg_func) for i in x]
    with open(out_directory+'means.csv','w') as f:
        f.write('leaning,label\n')
        for i,j in zip(true_x,y):
            f.write(str(i)+','+str(j)+'\n')

def generate_filenumbers(arr):
    '''
    This function generates the numbers that will be used for iterating through the IBC.
    '''
    output = []
    previous = None
    print(arr)
    for i in arr:
        if i == arr[-1]: #if you're at the end,
            output.append([previous, None]) #use None in order to ensure you reach the end of the list
            continue #leave the loop
        output.append([previous, i]) #use the previous number to avoid using the same data again
        previous = i #reset previous
    return output

def chunk_processor(filename,c,start,end,word_dict,m,temp_directory):
    '''
    This function processes 100 corpora from the Ideological Books Corpus
    It retrieves ratings from Ludovic Rheault and Christopher Cochrane's partyembed model
    With these ratings, it writes to tsv files numbered appropriately
    '''
    count = 0
    with open(temp_directory+filename, 'w') as f: #open the filename to which you'll write the 100-corpus chunk
        for i in c[start:end]: #for each corpus
            result = []
            x = i.get_words().lower().translate(str.maketrans('', '', string.punctuation)) # lowercase and remove punctuation
            for j in x.split(" "): #for each word
                word_dict, ratings = dict_updater(word_dict, m, j) #run dict_updater, update word_dict and get the ratings
                result.append(ratings) #append the rating
            f.write(str(count) + '\t' + str(result) + '\n') #write all ratings
            count += 1
    return word_dict

def dict_updater(d, m, word):
    '''
    Given a word, this function uses takes in a nested dictionary and adds the word
    with a rating given by Ludovic Rheault and Christopher Cochrane's partyembed model.
    The nested dictionary is set up such that for some word "example", it will be
    contained in the dictionary in:
    {'e': {'x': {'example': (leaning-dem, leaning-rep)}}}
    it returns the updated dictionary and the leanings.
    If the word is not in the vocabulary, 0 will be used.
    '''
    if len(word) < 2:
        return d, 0
    if word[0] in d.keys():
        if word[1] in d[word[0]].keys():
            if word in d[word[0]][word[1]].keys():
                return d, d[word[0]][word[1]][word]
            else:
                try:
                    issue = m.issue(word)[m.issue(word)['year'] == 2015]
                    d[word[0]][word[1]][word] = (issue['dem'], issue['rep'])
                    return d, d[word[0]][word[1]][word]
                except:
                    d[word[0]][word[1]][word] = 0
                    return d, 0
        else:
            try:
                issue = m.issue(word)[m.issue(word)['year'] == 2015]
                d[word[0]][word[1]] = {word: (issue['dem'], issue['rep'])}
                return d, d[word[0]][word[1]][word]
            except:
                d[word[0]][word[1]] = {word: 0}
                return d, 0
    else:
        try:
            issue = m.issue(word)[m.issue(word)['year'] == 2015]
            d[word[0]] = {word[1]: {word: (issue['dem'], issue['rep'])}}
            return d, d[word[0]][word[1]][word]
        except:
            d[word[0]] = {word[1]: {word: 0}}
            return d, 0
        
def combiner(prefix, number):
    '''
    This function converts a set of tsvs into labels and items.
    '''
    starter = ''
    labels = []
    items = []
    started = False
    for i in np.arange(100,number,100):
        title = prefix + '-' + str(i) + '.tsv' #create filename
        with open(title, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                if not started:
                    started = True
                    starter = line.split('[')[1].strip() + ' ' #account for first line special case
                    continue
                if (line[0] == 'N'): #if in the middle of a line
                    starter += line.strip() + ' ' #append to your current WIP
                else:
                    labels.append(prefix) #otherwise append everything
                    items.append(starter.replace('Name: dem, dtype: float64,', 'dem').replace('Name: rep, dtype: float64)', 'rep').replace('71    ','').replace('71   ','').replace('(',''))
                    starter = line.split('[')[1].strip() + ' ' #and restart
    return labels, items

def test_combiner(prefix, number):
    '''
    This function converts a set of tsvs into labels and items.
    '''
    starter = ''
    labels = []
    items = []
    started = False
    title = prefix + '-' + str(number) + '.tsv' #create filename
    with open('test/temp/' + title, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            if not started:
                started = True
                starter = line.split('[')[1].strip() + ' ' #account for first line special case
                continue
            if (line[0] == 'N'): #if in the middle of a line
                starter += line.strip() + ' ' #append to your current WIP
            else:
                labels.append(prefix) #otherwise append everything
                items.append(starter.replace('Name: dem, dtype: float64,', 'dem').replace('Name: rep, dtype: float64)', 'rep').replace('71    ','').replace('71   ','').replace('(',''))
                starter = line.split('[')[1].strip() + ' ' #and restart
    return labels, items

def func(nums, function = 'mean'):
    '''
    This function returns the aggregate function on the items
    '''
    fixed = []
    for i in nums: #extract numbers
        if len(i) > 5:
            d,r = i.split(' dem ')[0], i.split(' dem ')[1].split(' rep')[0]
            fixed.append(float(d) - float(r))
        else:
            fixed.append(0)
    if function == 'mean': #perform aggregate function
        return np.mean(fixed)
    elif function == 'max':
        return np.max(fixed)
    else:
        return np.min(fixed)
    
    
####################################################################################################
## The following code is not ours.                                                                ##
## https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url ##
####################################################################################################

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
