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
import requests
import json
import glob
import time

from src.etl.get_anames import scrape_anames, retrieve_anames
def get_all_history_stats(test = False):
    
    # Setup for string for cleaning
    stpwrds = stopwords.words("english")
    porter = stem.PorterStemmer()
    nltk.download("wordnet")
    nltk.download("stopwords")
    
    # Base directory for saving/processing article histories
    if test:
        xmls_base = "test/temp/wiki_xmls/"
    else:
        xmls_base = "src/data/temp/wiki_xmls/"
    
    if not os.path.exists(xmls_base):
        os.makedirs(xmls_base)
        
    # Base directory for saving resdicts
    rd_base = "src/data/temp/resdicts/"
    if not os.path.exists(rd_base):
        os.makedirs(rd_base)
    
    # Get and split anames into chunks of 20 for chunk-wise processing
    anames = retrieve_anames()
    alst = [anames[i:i + 20] for i in range(0, len(anames), 20)]
    
    # Load in the score dictionary
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
    
    # Helper for cleaning strings
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
        

    def get_art_hists(for_hist):
        for_hist_und = ["_".join(i.split()) for i in for_hist]
        exp_base = "https://en.wikipedia.org/w/index.php?title=Special:Export&pages="
        exp_end = "&history=1&action=submit"
        

        

        for tit in for_hist_und:
            url = exp_base + tit + exp_end
            try:
                resp = requests.get(url)
            except Exception as e:
                try:
                    time.sleep(10)
                    resp = requests.get(url)
                except Exception as e:
                    print(tit + " did not get processed")

            with open(xmls_base + tit + ".xml", mode = "wb") as wfile:
                wfile.write(resp.content)

            resp.close()
            
            
    def get_hist_stats(rdname):
        xmls_list = [x for x in os.listdir(xmls_base) if ".xml" in x]

        resdict = {}
        for fn in xmls_list:

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
            
        cnt = 0

        # Populate resdict with stats
        for name, revl in resdict.items():
            prevr = db.bigram(preproc_strn(revl[0]["text"])[0])
            for rev in revl:
#                 if cnt % 1000 == 1:
#                     print(cnt)
                cnt += 1
                curr = db.bigram(preproc_strn(rev["text"])[0])
                diffs = db.unique_items(prevr, curr)
                rem, add = diffs

                # Trying to get the following output: [absscore, sumscore, numwords, counts_list, totphrs]
                rem_abs = 0
                add_abs = 0
                rem_sum = 0
                add_sum = 0
                rem_num = len(rem)
                add_num = len(add)
        #         add_counts = {}
        #         rem_counts = {}
                add_phrs = 0
                rem_phrs = 0

                for bigr in rem:
                    if bigr in score_dict.keys():
                        rem_abs += abs(score_dict[bigr])
                        rem_sum += score_dict[bigr]
                        rem_phrs += 1

                for bigr in add:  
                    if bigr in score_dict.keys():
                        add_abs += abs(score_dict[bigr])
                        add_sum += score_dict[bigr]
                        add_phrs += 1

                rev["rem"] = rem
                rev["add"] = add
                rev["rem_abs"] = rem_abs
                rev["add_abs"] = add_abs
                rev["rem_sum"] = rem_sum
                rev["add_sum"] = add_sum
                rev["rem_num"] = rem_num
                rev["add_num"] = add_num
                rev["rem_phrs"] = rem_phrs
                rev["add_phrs"] = add_phrs

                del rev["text"]
                prevr = curr

        
        with open(rd_base + rdname + ".json", "w") as outfile:  
            json.dump(resdict, outfile) 
            
        
    def del_art_hists():
        files = glob.glob(xmls_base + "*")
        for f in files:
            os.remove(f)
            
            
    if not test:
        for ind, hst in enumerate(alst):
            # For each chunk of 20 scrapes, processes, and deletes the articles
            get_art_hists(hst)
            get_hist_stats("rd" + str(ind+1))
            del_art_hists()
    else:
        get_hist_stats("testrd")

        
        

