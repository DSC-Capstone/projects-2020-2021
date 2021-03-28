import pickle
import nltk
from nltk import stem
from nltk.corpus import stopwords
import string
import src.data.full_ibc.treeUtil as treeUtil
import json

def get_scores():
    [lib, con, neutral] = pickle.load(open('src/data/full_ibc/ibcData.pkl', 'rb'))

    # Parse IBC into lists for each bias type (liberal, conservative, neutral)

    liblist = []
    for tree in lib:
        liblist.append(tree.get_words())
        
    conlist = []
    for tree in con:
        conlist.append(tree.get_words())
        
    neutlist = []
    for tree in neutral:
        neutlist.append(tree.get_words())

    nltk.download("wordnet")
    nltk.download("stopwords")

    # Helper for text pre-processing: lowercase, removing punctuation and stop-words, stemming
    def process_sentlist(sentlist):
        proc_sentlist = []
        for sent in sentlist:
            # Lowercase and remove punctuation
            curstr = sent.lower().translate(str.maketrans('', '', string.punctuation))

            plst = []
            for word in curstr.split():
                # Check for stopwords
                if word not in stpwrds:
                    # Stem the word
                    pword = porter.stem(word)

                    # Put the stemmed word in the reverse-stemming dictionary
                    if pword not in stem_rev.keys():
                        stem_rev[pword] = [word]
                    elif word not in stem_rev[pword]:
                        stem_rev[pword].append(word)

                    plst.append(pword)

            curstr = ' '.join(plst)
            proc_sentlist.append(curstr)
        return proc_sentlist


    # Helper for getting word frequency dictionaries for word combinations of length l
    def get_freq_dict(proc_list, l):
        freq_dict = {}
        for sent in proc_list:
            sentlist = sent.split()
            for i in range(len(sentlist)-l):
                curphr = " ".join(sentlist[i:i+l])
                if curphr not in freq_dict.keys():
                    freq_dict[curphr] = 1
                else:
                    freq_dict[curphr] += 1
        return freq_dict


    # Helper for getting x2 scores from the first cnt entries of the frequency dictionaries
    def get_x2(dict1, dict2, cnt = -1):
        res_dict = {}
        # If cnt = -1 use full dictionaries
        if cnt == -1:
            d1 = dict1
            d2 = dict2
        else:
            # Sort and get the first cnt entries
            d1 = dict(sorted(dict1.items(), key=lambda item: item[1], reverse=True)[:cnt])
            d2 = dict(sorted(dict2.items(), key=lambda item: item[1], reverse=True)[:cnt])
            
        all_phrases = list(set(list(d1.keys()) + list(d2.keys())))
        d1_sum = sum(list(d1.values()))
        d2_sum = sum(list(d2.values()))
        
        for phrase in all_phrases:
            # Get values
            if phrase not in d1.keys():
                f1 = 0
            else:
                f1 = d1[phrase]
                
            if phrase not in d2.keys():
                f2 = 0
            else:
                f2 = d2[phrase]
                
            f1c = d1_sum - f1
            f2c = d2_sum - f2
            
            # Plug the values into the formula
            curx2 = (f1*f2c - f2*f1c)**2 / ((f1 + f2)*(f1+f1c)*(f2+f2c)*(f1c+f2c))
            res_dict[phrase] = curx2
            
        return res_dict


    # Dictionary for reverse-stemming
    stem_rev = {}
    # Get list of stopwords
    stpwrds = stopwords.words("english")
    porter = stem.PorterStemmer()


    proc_liblist = process_sentlist(liblist)
    proc_conlist = process_sentlist(conlist)
    # proc_biaslist = proc_liblist + proc_conlist
    # proc_neutlist = process_sentlist(neutlist) 

        
    lib_freq2 = get_freq_dict(proc_liblist, 2)
    lib_freq3 = get_freq_dict(proc_liblist, 3)
    con_freq2 = get_freq_dict(proc_conlist, 2)
    con_freq3 = get_freq_dict(proc_conlist, 3)
    # neut_freq2 = get_freq_dict(proc_neutlist, 2)
    # neut_freq3 = get_freq_dict(proc_neutlist, 3)
    # bias_freq2 = get_freq_dict(proc_biaslist, 2)
    # bias_freq3 = get_freq_dict(proc_biaslist, 3)
        


    x2_libcon2 = get_x2(lib_freq2, con_freq2, 1000)
    x2_libcon3 = get_x2(lib_freq3, con_freq3, 1000)


    # Dump reverse-stemming dictionary into JSON file
    with open('src\data\stem_rev.json', 'w') as fp:
        json.dump(stem_rev, fp)

    # Dump dictionaries into JSON files
    with open('src\data\scores_2.json', 'w') as fp:
        json.dump(x2_libcon2, fp)
    with open('src\data\scores_3.json', 'w') as fp:
        json.dump(x2_libcon3, fp)

    return x2_libcon2, x2_libcon3

def string_score(strn, score_dict):
    curstr = strn.lower().translate(str.maketrans('', '', string.punctuation))

    plst = []
    for word in curstr.split():
        # Check for stopwords
        if word not in stpwrds:
            # Stem the word
            pword = porter.stem(word)

    #         # Put the stemmed word in the reverse-stemming dictionary
    #         if pword not in stem_rev.keys():
    #             stem_rev[pword] = [word]
    #         elif word not in stem_rev[pword]:
    #             stem_rev[pword].append(word)

            plst.append(pword)
    curstrlen = len(plst)
    curstr = ' '.join(plst)

    curscore = 0

    for key, value in score_dict.items():
        curscore += curstr.count(key)*value

    return curscore / curstrlen
    