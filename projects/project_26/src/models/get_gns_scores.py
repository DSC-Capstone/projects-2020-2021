import re
import os
import nltk
from nltk import stem
from nltk.corpus import stopwords
import string

def get_stat_dict(nametxt_dict, test=False):

    if test:
        pp_dir = "test/partisan_phrases/"
    else:
        pp_dir = "src/data/init/partisan_phrases/"
    pp_txts = os.listdir(pp_dir)
    score_dict = {}
    for i in pp_txts:
        with open(pp_dir + i) as curtxt:
            for line in curtxt.readlines()[1:]:
                splt = line.split("|")
                score_dict[splt[0]] = float(splt[1].strip())
        
    nltk.download("wordnet")
    nltk.download("stopwords")

    stpwrds = stopwords.words("english")
    porter = stem.PorterStemmer()

    def preproc_strn(strn):
        # Lowercase, remove digits and doublespaces
        curstr = strn.lower().translate(str.maketrans('', '', string.punctuation))
        curstr = re.sub(r'[0-9]+', '', curstr)
        curstr = re.sub(r'\n', ' ', curstr)
        curstr = re.sub(r'  +', ' ', curstr)
        plst = []
        for word in curstr.split():
            # Check for stopwords
            if word not in stpwrds:
                # Porter stem the word
                pword = porter.stem(word)
                plst.append(pword)
        numwords = len(plst)
        curstr = ' '.join(plst)
        return (curstr, numwords)

    def string_score(strn, score_dict):
        # Pre-process, return the processed string and the number of words
        curstr, numwords = preproc_strn(strn)

        # Absolute bias sum
        absscore = 0
        # Bias sum
        sumscore = 0
        # Total number of occurences of phrases from G&S
        totphrs = 0
        
        # Dictionary of top 10 phrase counts
        counts_dict = {}
        
        for key, value in score_dict.items():
            
            numoccurs = curstr.count(key)
            totphrs += numoccurs
            counts_dict[key] = (numoccurs, value)
            curscore = numoccurs*value
            absscore += abs(curscore)
            sumscore += curscore

        counts_list = sorted(counts_dict.items(), key=lambda item: item[1], reverse=True)[:10]
        return [absscore, sumscore, numwords, counts_list, totphrs]

    namestat_dict = {}

    for name, txt in nametxt_dict.items():
        namestat_dict[name] = string_score(txt, score_dict)
        
    for name, stat in namestat_dict.items():
        dispcnt = 1
        procname = preproc_strn(name)[0]
        is_intitle = False
        for phr, freq in stat[3]:
            if phr in procname:
                is_intitle = True
            dispcnt += 1

        namestat_dict[name].append(is_intitle)
    return namestat_dict