import src.models.difflib_bigrams as db
import os
import pandas as pd
from nltk.corpus import stopwords
from nltk import stem
import nltk
import xml.etree.ElementTree as ET
import string
import re
import numpy as np
from partyembed.explore import Explore
from partyembed.utils import issues

def main():
    '''
    Main function for getting political leanings with
    partyembed on revision histories.
    '''
    stpwrds = stopwords.words("english")

    nltk.download("wordnet")
    nltk.download("stopwords")

    def preproc_strn(strn):
        '''
        Preprocesses the string, removing punctuation, digits,
        doublespaces, and stopwords.
        '''
        # Lowercase, remove digits and doublespaces
        curstr = strn.lower().translate(str.maketrans('', '', string.punctuation))
        curstr = re.sub(r'[0-9]+', '', curstr)
        curstr = re.sub(r'\n', ' ', curstr)
        curstr = re.sub(r'  +', ' ', curstr)
        plst = []
        for word in curstr.split():
            # Check for stopwords
            if word not in stpwrds:
                plst.append(word)
        numwords = len(plst)
        curstr = ' '.join(plst)
        return curstr

    xmls_base = "data/temp/wiki_xmls/"
    xmls_list = [x for x in os.listdir(xmls_base) if ".xml" in x]

    resdict = {}
    for fn in xmls_list:
        print(fn)

        # This block is for fixing broken xmls with no closing tags
        try:
            tree = ET.parse(xmls_base + fn)
        except Exception as e:
            with open(xmls_base + fn, "a") as app:
                app.write("  </page>")
                app.write("</mediawiki>")
            tree = ET.parse(xmls_base + fn)

        # Set up the tree and the list of results for the current article
        root = tree.getroot().find("{http://www.mediawiki.org/xml/export-0.10/}page")
        revlist = []

        for rev in root.findall("{http://www.mediawiki.org/xml/export-0.10/}revision"):
            # The dictionary for each revision
            curdict = {}

            curdict["time"] = rev.find("{http://www.mediawiki.org/xml/export-0.10/}timestamp").text
            txt = rev.find("{http://www.mediawiki.org/xml/export-0.10/}text").text

            if not txt is None:
                curdict["text"] = txt
            else:
                curdict["text"] = ""

            comm = rev.find("{http://www.mediawiki.org/xml/export-0.10/}comment")
            if not comm is None:
                curdict["comm"] = comm.text
            else:
                curdict["comm"] = ""

            cont = rev.find("{http://www.mediawiki.org/xml/export-0.10/}contributor")
            user = cont.find("{http://www.mediawiki.org/xml/export-0.10/}username")
            if not user is None:
                curdict["user"] = user.text
            else:
                curdict["user"] = cont.find("{http://www.mediawiki.org/xml/export-0.10/}ip").text

            revlist.append(curdict)

        resdict[fn[:-4]] = revlist

    m = Explore(model = 'House') #initiate partyembed model

    def smaller_ttd(textlist):
        '''
        This function is essentially text_to_dict on smaller subsets
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        return output

    def text_to_dict(textlist):
        '''
        converts a split text into a dictionary
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        for i in np.arange(10): #delete top 10 words
            del output[max(output, key=output.get)]
        return output

    def outer(article_dict,m):
        '''
        This function takes in a dictionary representation of an article
        (represented by text_to_dict(textlist)) and outputs the mean
        democratic leaning and mean republican leaning
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
        results = 0
        divide = 0
        for key, value in article_dict.items():
            try:
                demo, repu = inner(key,m,z,parties)
                diff = demo - repu
                for k in np.arange(value):
                    results += diff
                    divide += 1
            except:
                continue
        return results, divide

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


    for title,i in resdict.items():
        count = 0
        lines = []

        prev = preproc_strn(i[0]['text']).split(" ") #clean the initial version of the article
        initial = outer(text_to_dict(prev),m) #and get its initial leaning
        lines.append(title + "\t" + i[0]['time'] + "\t" + str(initial[0]/initial[1])) #write a line account for this
        newsum, newdivide = initial[0], initial[1]
        for j in i[1:]: #for each subsequent revision,
            count += 1
            curr = preproc_strn(j['text']).split(" ") #clean and split the new version
            former, latter = db.unique_items(prev, curr) #find what words were removed and added
            subtract = outer(smaller_ttd(former),m) #recieve scores adjustments for the removed words
            add = outer(smaller_ttd(latter),m) #and the added words
            #subtract and add respectively
            newsum, newdivide = newsum - subtract[0] + add[0], newdivide - subtract[1] + add[1]
            prev = curr #update the frame of reference
            lines.append(title + "\t" + j['time'] + "\t" + str(newsum/newdivide)) #add line
            if count % 200 == 0: #in increments of 200, add information to a file
                filename = 'data/output/' + title + str(count) + '.tsv'
                print(filename)
                with open(filename,'w') as f:
                    for line in lines:
                        f.write(line + '\n')
                lines = []
        filename = 'data/output/' + title + str(count) + '.tsv'
        with open(filename,'w') as f: #account for the remainder (after the final 200 increment)
            for line in lines:
                f.write(line + '\n')
                
def test():
    '''
    test function for getting political leanings with
    partyembed on revision histories.
    '''
    stpwrds = stopwords.words("english")

    nltk.download("wordnet")
    nltk.download("stopwords")

    def preproc_strn(strn):
        '''
        Preprocesses the string, removing punctuation, digits,
        doublespaces, and stopwords.
        '''
        # Lowercase, remove digits and doublespaces
        curstr = strn.lower().translate(str.maketrans('', '', string.punctuation))
        curstr = re.sub(r'[0-9]+', '', curstr)
        curstr = re.sub(r'\n', ' ', curstr)
        curstr = re.sub(r'  +', ' ', curstr)
        plst = []
        for word in curstr.split():
            # Check for stopwords
            if word not in stpwrds:
                plst.append(word)
        numwords = len(plst)
        curstr = ' '.join(plst)
        return curstr

    xmls_base = "test/temp/wiki_xmls/"
    xmls_list = [x for x in os.listdir(xmls_base) if ".xml" in x]

    resdict = {}
    for fn in xmls_list:
        print(fn)

        # This block is for fixing broken xmls with no closing tags
        try:
            tree = ET.parse(xmls_base + fn)
        except Exception as e:
            with open(xmls_base + fn, "a") as app:
                app.write("  </page>")
                app.write("</mediawiki>")
            tree = ET.parse(xmls_base + fn)

        # Set up the tree and the list of results for the current article
        root = tree.getroot().find("{http://www.mediawiki.org/xml/export-0.10/}page")
        revlist = []

        for rev in root.findall("{http://www.mediawiki.org/xml/export-0.10/}revision"):
            # The dictionary for each revision
            curdict = {}

            curdict["time"] = rev.find("{http://www.mediawiki.org/xml/export-0.10/}timestamp").text
            txt = rev.find("{http://www.mediawiki.org/xml/export-0.10/}text").text

            if not txt is None:
                curdict["text"] = txt
            else:
                curdict["text"] = ""

            comm = rev.find("{http://www.mediawiki.org/xml/export-0.10/}comment")
            if not comm is None:
                curdict["comm"] = comm.text
            else:
                curdict["comm"] = ""

            cont = rev.find("{http://www.mediawiki.org/xml/export-0.10/}contributor")
            user = cont.find("{http://www.mediawiki.org/xml/export-0.10/}username")
            if not user is None:
                curdict["user"] = user.text
            else:
                curdict["user"] = cont.find("{http://www.mediawiki.org/xml/export-0.10/}ip").text

            revlist.append(curdict)

        resdict[fn[:-4]] = revlist

    m = Explore(model = 'House') #initiate partyembed model

    def smaller_ttd(textlist):
        '''
        This function is essentially text_to_dict on smaller subsets
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        return output

    def text_to_dict(textlist):
        '''
        converts a split text into a dictionary
        '''
        output = {}
        for i in textlist:
            try:
                output[i] += 1
            except:
                output[i] = 1
        for i in np.arange(10): #delete top 10 words
            del output[max(output, key=output.get)]
        return output

    def outer(article_dict,m):
        '''
        This function takes in a dictionary representation of an article
        (represented by text_to_dict(textlist)) and outputs the mean
        democratic leaning and mean republican leaning
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
        results = 0
        divide = 0
        for key, value in article_dict.items():
            try:
                demo, repu = inner(key,m,z,parties)
                diff = demo - repu
                for k in np.arange(value):
                    results += diff
                    divide += 1
            except:
                continue
        return results, divide

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


    for title,i in resdict.items():
        count = 0
        lines = []

        prev = preproc_strn(i[0]['text']).split(" ") #clean the initial version of the article
        initial = outer(text_to_dict(prev),m) #and get its initial leaning
        lines.append(title + "\t" + i[0]['time'] + "\t" + str(initial[0]/initial[1])) #write a line account for this
        newsum, newdivide = initial[0], initial[1]
        for j in i[1:25]: #for each subsequent revision,
            count += 1
            curr = preproc_strn(j['text']).split(" ") #clean and split the new version
            former, latter = db.unique_items(prev, curr) #find what words were removed and added
            subtract = outer(smaller_ttd(former),m) #recieve scores adjustments for the removed words
            add = outer(smaller_ttd(latter),m) #and the added words
            #subtract and add respectively
            newsum, newdivide = newsum - subtract[0] + add[0], newdivide - subtract[1] + add[1]
            prev = curr #update the frame of reference
            lines.append(title + "\t" + j['time'] + "\t" + str(newsum/newdivide)) #add line
            if count % 20 == 0: #in increments of 200, add information to a file
                filename = 'test/output/' + title + str(count) + '.tsv'
                print(filename)
                with open(filename,'w') as f:
                    for line in lines:
                        f.write(line + '\n')
                lines = []