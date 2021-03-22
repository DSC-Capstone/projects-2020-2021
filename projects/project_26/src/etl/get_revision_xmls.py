import requests
import os

def main():
    '''
    This function obtains the articles of interest for analyzing bias over time
    '''
    
    for_hist = ["Era of Good Feelings", "Late-night talk show", "James K. Polk", "Jim Acosta", "Separation of church and state in the United States", "Mueller report", "Justice Democrats","Tammy Baldwin","Democratic Party (United States)"]
    for_hist_und = ["_".join(i.split()) for i in for_hist]
    exp_base = "https://en.wikipedia.org/w/index.php?title=Special:Export&pages="
    exp_end = "&history=1&action=submit"
    xml_base = 'data/temp/wiki_xmls/'

    if not os.path.exists(xml_base):
        os.makedirs(xml_base)

    cnt = 1
    for tit in for_hist_und:
        url = exp_base + tit + exp_end #build URL
        resp = requests.get(url) #request

        with open(xml_base + tit + ".xml", mode = "wb") as wfile:
            wfile.write(resp.content) #write file

        resp.close()
        print(str(cnt) + " processed")
        cnt += 1
        
def test():
    '''
    This function obtains the articles of interest for analyzing bias over time
    '''
    
    for_hist = ["Jim Acosta"]
    for_hist_und = ["_".join(i.split()) for i in for_hist]
    exp_base = "https://en.wikipedia.org/w/index.php?title=Special:Export&pages="
    exp_end = "&history=1&action=submit"
    xml_base = 'test/temp/wiki_xmls/'

    if not os.path.exists(xml_base):
        os.makedirs(xml_base)

    cnt = 1
    for tit in for_hist_und:
        url = exp_base + tit + exp_end #build URL
        resp = requests.get(url) #request

        with open(xml_base + tit + ".xml", mode = "wb") as wfile:
            wfile.write(resp.content) #write file

        resp.close()
        print(str(cnt) + " processed")
        cnt += 1