import requests
from bs4 import BeautifulSoup
import os
from src.libcode import list_to_txt, txt_to_list

def scrape_anames(fname="artnames.txt"):
    fbase = "src/data/temp/"
    if not os.path.exists(fbase):
        os.makedirs(fbase)
    fpath = fbase + fname
#     # request social and political philosophy index
#     spp_resp = requests.get("https://en.wikipedia.org/wiki/Index_of_social_and_political_philosophy_articles")
#     # request political index
#     pol_resp = requests.get("https://en.wikipedia.org/wiki/Index_of_politics_articles")
    # request american politics list
    ampol = requests.get("https://en.wikipedia.org/wiki/Wikipedia:WikiProject_Politics/American_politics#List")

    # soupify
#     spp_soup = BeautifulSoup(spp_resp.text)
#     pol_soup = BeautifulSoup(pol_resp.text)
    ampol_soup = BeautifulSoup(ampol.text)
    # make a combined list of article names
    fin_list = []

#     # the bs4 part is hardcoded and quite arbitrary, if wikipedia structure changes this code won't work
#     for i in spp_soup.find_all("ul")[1:26]:
#         for j in i.find_all("a"):
#             fin_list.append(j.get("title"))
            
#     for i in pol_soup.find_all("p")[1:]:
#         for j in i.find_all("a"):
#             fin_list.append(j.get("title"))
            
    ampol_tbl = ampol_soup.find_all("table", attrs="wikitable sortable")[0]
    td_list = ampol_tbl.find_all("td")
    for i in range(len(td_list)):
        if (i+1) % 6 == 2:
            a_name = td_list[i].find("a")["title"]
            a_cnt = td_list[i+1].text
            a_class = td_list[i+3].text
            if "list" not in a_name.lower():
                fin_list.append(a_name)
       
#     # remove duplicates     
#     fin_list = list(set(fin_list))

    # write to file
    list_to_txt(fpath, fin_list)

    return fname

def retrieve_anames(fname="artnames.txt"):
    fpath = "src/data/temp/" + fname
    return txt_to_list(fpath)