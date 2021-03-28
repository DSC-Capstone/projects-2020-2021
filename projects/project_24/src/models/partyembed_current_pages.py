from partyembed.explore import Explore
import pandas as pd
from partyembed.utils import issues
import numpy as np
import os
import zipfile
import string
from nltk.corpus import stopwords

def main():
    '''
    This function analyzes our chosen political current-page articles 
    and outputs .tsv files containing the article name and overall bias
    '''
    if not os.path.exists('data/output/'): #create output directory
        os.makedirs('data/output/')
    
    m = Explore(model = 'House') #initial partyembed model

    titles = []
    articles = []
    count=0

    for i in np.arange(1,11):
        with open('data/wiki_txts/art_pages' + str(i) + '.txt', 'r') as f:
            first = True #there's a special case for the first article in a txt file
            while True:
                line = f.readline()
                if not line: #at the end of a file
                    articles.append(article) #add the article you finished
                    break
                if '~!~' in line: #article delimiter, surrounds title
                    sp = line.split("~!~")
                    if (len(sp) != 3): #in this case, it's just the title and the beginning of an article
                        if first:
                            first = False #in the first case you just add the title and begin the article
                            titles.append(sp[0])
                            article = sp[1].strip()
                        else: #otherwise, you have an finished article to add
                            articles.append(article)
                            titles.append(sp[0])
                            article = sp[1].strip()
                    else: #in this scenario, there's a bit of the last article the beginning
                        article += sp[0] #finish it, append, and begin the next title+article
                        articles.append(article)
                        titles.append(sp[1])
                        article = sp[2].strip()
                else: #otherwise, just add to the article you're building
                    article += line.strip()

    def text_to_dict(textlist):
        '''
        This function takes in a text split by " " characters and
        turns it into a dictionary with entries Word: # occurrences
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        newdict = {}
        for i,j in output.items():
            if i not in stopwords.words('english'):
                newdict[i]=j
        for i in np.arange(10): #remove top 10 most common phrases
            del newdict[max(newdict, key=newdict.get)]
        return newdict

    def outer(article_dict,m):
        '''
        This function takes in a dictionary representation of an article
        (represented by text_to_dict(textlist)) and outputs the overall
        political leaning of an article, with 1 being democratic and -1
        being republican
        '''
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XX The following code is taken from the source code of Rheault    XX
        # XX and Cochrane's partyembed. This is here in order to circumvent XX
        # XX the creation of a pandas dataframe for improved speed. For     XX
        # XX the full code, see:                                            XX    
        # XX                                                                XX
        # XX https://github.com/lrheault/partyembed                         XX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        M = m.model.vector_size
        dv = m.model.docvecs.offset2doctag
        parliaments = [i for i in range(43,115)]
        years = [i for i in range(1873,2017,2)]
        parties = [d for d in dv if d.startswith('D_') or d.startswith('R_')]
        P = len(parties)
        z = np.zeros(( P, M ))
        for i in range(P):
            z[i,:] = m.model.docvecs[parties[i]]
        results = []
        for key, value in article_dict.items():
            try:
                demo, repu = inner(key,m,z,parties)
                diff = (demo-repu)*value
                results.append(diff)
            except:
                continue
        return np.mean(results)

    def inner(word,m,z,parties):
        '''
        Additional code for outer(), calculates mean democratic and republican
        leanings from 2007 onwards
        '''
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XX The following code is taken from the source code of Rheault    XX
        # XX and Cochrane's partyembed. This is here in order to circumvent XX
        # XX the creation of a pandas dataframe for improved speed. For     XX
        # XX the full code, see:                                            XX    
        # XX                                                                XX
        # XX https://github.com/lrheault/partyembed                         XX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        topvec = issues.bootstrap_topic_vector(word, m.model, n = 2, sims=1)
        C, LB, UB = issues.cos_sim(z, topvec, boot=True, sims=1)
        dem = [s for t,s in zip(parties,C) if t.startswith('D_')]
        rep = [s for t,s in zip(parties,C) if t.startswith('R_')]
        return np.mean(dem[-5:]), np.mean(rep[-5:])


    def writer(number):
        '''
        For writing the .tsv results in increments of 50
        '''
        count = number - 50 #set the count for which article you are on, modified
        index = number - 50 #for setting the loop index, not to be modified
        filename = 'data/output/result' + str(number) + '.tsv'
        
        with open(filename,'w') as f:
            for i,j in zip(titles[index:number],articles[index:number]):
                results = []
                text = j.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
                text = text.lower() #lower
                textd = text_to_dict(text.split(" ")) #split, get counts
                mean = outer(textd,m) #get slants
                print(str(count) + ": " + i + ": " + str(mean)) #for visual progress
                f.write(str(count) + '\t' + i + '\t' + str(mean) + '\n') #write line
                count += 1 #increment article

    def final_writer():
        '''
        As the articles aren't evenly divisible by 50, this is for the last batch
        '''
        count = 600 
        filename = 'data/output/result' + str(len(titles)) + '.tsv'
        with open(filename,'w') as f:
            for i,j in zip(titles[600:],articles[600:]):
                results = []
                text = j.translate(str.maketrans('', '', string.punctuation))
                text = text.lower()
                textd = text_to_dict(text.split(" "))
                mean = outer(textd,m)
                print(str(count) + ": " + i + ": " + str(mean))
                f.write(str(count) + '\t' + i + '\t' + str(mean) + '\n')
                count += 1

    for k in np.arange(50,650,50):
        writer(k)
    final_writer()

def test():
    '''
    A test function, running on a smaller subset of the data
    '''
    if not os.path.exists('test/output/'): #create output directory
        os.makedirs('test/output/')
    
    m = Explore(model = 'House') #initial partyembed model

    titles = []
    articles = []
    count=0

    with open('test/wiki_txts/art_pages1.txt', 'r') as f:
        first = True #there's a special case for the first article in a txt file
        while True:
            line = f.readline()
            if not line: #at the end of a file
                articles.append(article) #add the article you finished
                break
            if '~!~' in line: #article delimiter, surrounds title
                sp = line.split("~!~")
                if (len(sp) != 3): #in this case, it's just the title and the beginning of an article
                    if first:
                        first = False #in the first case you just add the title and begin the article
                        titles.append(sp[0])
                        article = sp[1].strip()
                    else: #otherwise, you have an finished article to add
                        articles.append(article)
                        titles.append(sp[0])
                        article = sp[1].strip()
                else: #in this scenario, there's a bit of the last article the beginning
                    article += sp[0] #finish it, append, and begin the next title+article
                    articles.append(article)
                    titles.append(sp[1])
                    article = sp[2].strip()
            else: #otherwise, just add to the article you're building
                article += line.strip()

    def text_to_dict(textlist):
        '''
        This function takes in a text split by " " characters and
        turns it into a dictionary with entries Word: # occurrences
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        newdict = {}
        for i,j in output.items():
            if i not in stopwords.words('english'):
                newdict[i]=j
        for i in np.arange(10): #remove top 10 most common phrases
            del newdict[max(newdict, key=newdict.get)]
        return newdict

    def outer(article_dict,m):
        '''
        This function takes in a dictionary representation of an article
        (represented by text_to_dict(textlist)) and outputs the overall
        political leaning of an article, with 1 being democratic and -1
        being republican
        '''
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XX The following code is taken from the source code of Rheault    XX
        # XX and Cochrane's partyembed. This is here in order to circumvent XX
        # XX the creation of a pandas dataframe for improved speed. For     XX
        # XX the full code, see:                                            XX    
        # XX                                                                XX
        # XX https://github.com/lrheault/partyembed                         XX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        M = m.model.vector_size
        dv = m.model.docvecs.offset2doctag
        parliaments = [i for i in range(43,115)]
        years = [i for i in range(1873,2017,2)]
        parties = [d for d in dv if d.startswith('D_') or d.startswith('R_')]
        P = len(parties)
        z = np.zeros(( P, M ))
        for i in range(P):
            z[i,:] = m.model.docvecs[parties[i]]
        results = []
        for key, value in article_dict.items():
            try:
                demo, repu = inner(key,m,z,parties)
                diff = (demo-repu)*value
                results.append(diff)
            except:
                continue
        return np.mean(results)

    def inner(word,m,z,parties):
        '''
        Additional code for outer(), calculates mean democratic and republican
        leanings from 2007 onwards
        '''
        
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        # XX The following code is taken from the source code of Rheault    XX
        # XX and Cochrane's partyembed. This is here in order to circumvent XX
        # XX the creation of a pandas dataframe for improved speed. For     XX
        # XX the full code, see:                                            XX    
        # XX                                                                XX
        # XX https://github.com/lrheault/partyembed                         XX
        # XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        
        topvec = issues.bootstrap_topic_vector(word, m.model, n = 2, sims=1)
        C, LB, UB = issues.cos_sim(z, topvec, boot=True, sims=1)
        dem = [s for t,s in zip(parties,C) if t.startswith('D_')]
        rep = [s for t,s in zip(parties,C) if t.startswith('R_')]
        return np.mean(dem[-5:]), np.mean(rep[-5:])


    def writer(number):
        '''
        For writing the .tsv results in increments of 2
        '''
        count = number - 1 #set the count for which article you are on, modified
        index = number - 1 #for setting the loop index, not to be modified
        filename = 'test/output/result' + str(number) + '.tsv'
        
        with open(filename,'w') as f:
            for i,j in zip(titles[index:number],articles[index:number]):
                results = []
                text = j.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
                text = text.lower() #lower
                textd = text_to_dict(text.split(" ")) #split, get counts
                mean = outer(textd,m) #get slants
                print(str(count) + ": " + i + ": " + str(mean)) #for visual progress
                f.write(str(count) + '\t' + i + '\t' + str(mean) + '\n') #write line
                count += 1 #increment article

    writer(1)