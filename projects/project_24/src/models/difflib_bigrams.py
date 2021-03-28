import difflib
import string
import numpy as np

def bigram(text):
    '''
    This function takes a text, removes punctuation, and returns a list of all bigrams from the text
    '''
    #remove punctuation
    s = text.translate(str.maketrans('', '', string.punctuation))
    #split text into words
    l = s.split(" ")
    
    bigrams = []
    
    #for every word but the last,
    for i in np.arange(len(l)-1):
        #append to the list that word and the word following
        bigrams.append(l[i] + " " + l[i+1])
        
    return bigrams

def unique_items(bg1, bg2):
    '''
    Given two lists of bigrams, this function returns:
    1) Prior, a list of bigrams present in the first but not the second
    2) New, a list of bigrams present in the second but not the first
    '''
    
    prior = []
    new = []
    
    matcher = difflib.SequenceMatcher(None, bg1, bg2) #use difflib sequence matcher
    
    #this will go through operations for turning the former list into the latter
    for tag, i1, i2, j1, j2 in reversed(matcher.get_opcodes()):
        
        #when deleting from the former, this means the bigram is unique to it
        #add to Prior
        if tag == 'delete':
            for i in bg1[i1:i2]:
                prior.append(i)
        
        #when equal, there is no action necessary
        elif tag == 'equal':
            continue
            
        #when inserting into the former, this new bigram must be unique to the latter
        #add to New
        elif tag == 'insert':
            for i in bg2[j1:j2]:
                new.append(i)
        
        #when replacing, the replacer is unique to latter, but replaced is unique to former
        #add the replacer to new, and the replaced to prior
        elif tag == 'replace':
            for i in bg1[i1:i2]:
                prior.append(i)
            for i in bg2[j1:j2]:
                new.append(i)
                
    return prior, new