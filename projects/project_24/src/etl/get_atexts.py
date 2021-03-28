import wikipediaapi as wp
import os
import time
from src.libcode import txt_to_list, list_to_txt
from src.etl.get_anames import retrieve_anames

def scrape_atexts(test = False):
    enwp = wp.Wikipedia("en")
    anames = retrieve_anames()
    num_per_text = len(anames)//10
    if test:
        wikitxts_dir = "test/wiki_txts/"
    else:
        wikitxts_dir = "src/data/temp/wiki_txts/"

    # If wiki texts folder does not exist make it
    if not os.path.exists(wikitxts_dir):
        os.makedirs(wikitxts_dir)

    txtlst = []
    for ind, aname in enumerate(anames):
        # Get the page text
    #     print(ind, aname)
        try:
            curpg = enwp.page(aname)
            curtit = curpg.title
            curtxt = curpg.text
            txtlst.append(curtit)
            txtlst.append(curtxt)
        except Exception as e:
            print("FINALLY CAUGHT YOU, ", e)
            time.sleep(20)
            curpg = enwp.page(aname)
            curtit = curpg.title
            curtxt = curpg.text
            txtlst.append(curtit)
            txtlst.append(curtxt)
        # This ensures it saves into 10 txt files
        if (ind+1) % num_per_text == 0:
            print(ind+1)
            curtxt_name = "art_pages" + str((ind+1)//num_per_text) + ".txt"
            list_to_txt(wikitxts_dir+curtxt_name, txtlst)
            txtlst = []
            time.sleep(10)
            
    # Save last set of articles
    if len(txtlst) > 0:
        curtxt_name = "art_pages10" + ".txt"
        list_to_txt(wikitxts_dir+curtxt_name, txtlst)


def retrieve_atexts(test = False):
    if test:
        wikitxts_dir = "test/wiki_txts/"
    else:
        wikitxts_dir = "src/data/temp/wiki_txts/"
        
    numtxts = len(os.listdir(wikitxts_dir))
    wiki_txts = ["art_pages" + str(i) + ".txt" for i in range(1,numtxts+1)]
    nametxt_dict = {}

    cnt = 0
    for txt in wiki_txts:
        # print(cnt)
        txtlst = txt_to_list(wikitxts_dir + txt)
        for item in txtlst:
            if cnt % 2 == 0:
                aname = item
            else:
                nametxt_dict[aname] = item
            cnt += 1
    return nametxt_dict
