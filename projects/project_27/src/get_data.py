from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import math
import csv 
from bs4 import BeautifulSoup
import bz2
import lxml
import requests
import urllib.request
import zipfile


def find_count(name,name_dict):
    name_dict[name]+=1
    return name_dict[name]-1

def getMtest(url, to_csvname):     
    URL = url 
    r = requests.get(URL, allow_redirects=True)
    open(to_csvname, 'wb').write(r.content)


def download_xml_file(url, to_csvfilename):

    URL = url 
    filename = os.path.basename(url)
    urllib.request.urlretrieve(url,filename)
    newfilepath = filename[:-4]
    a1=open(newfilepath,'wb')
    with bz2.BZ2File(filename, "r") as file:
        for line in file:
            a1.write(line)
    a1.close()

  #parse the xml file 
  #convert the raw xml to pd.dataframe
    revision_list=[]
    a=0
    title_list=[]
    for event, elem in ET.iterparse(newfilepath,events=('end',)):
        if elem.tag=='{http://www.mediawiki.org/xml/export-0.10/}title':
            title_list.append(elem.text)
        if elem.tag=='{http://www.mediawiki.org/xml/export-0.10/}revision':
            each_revision={'Title':title_list[-1]}
            #find contributor of every revision
            contributor=elem.find('{http://www.mediawiki.org/xml/export-0.10/}contributor')
            try:
                username=contributor.find('{http://www.mediawiki.org/xml/export-0.10/}username').text
            except:
                username=np.nan
            each_revision['Contributor_Name']=username

            #find each revision time
            time=elem.find('{http://www.mediawiki.org/xml/export-0.10/}timestamp').text
            each_revision['time']=time

            #find each revision comment
            try:
                comment=elem.find('{http://www.mediawiki.org/xml/export-0.10/}comment').text
            except:
                comment=np.nan
            each_revision['comment']=comment 

            revision_list.append(each_revision)

            if len(revision_list)>=90000:
                df=pd.DataFrame(revision_list)
                df.to_csv('raw_dump/raw_dump_b'+str(a)+'.csv')
                revision_list=[]
                a+=1
            elem.clear()

    df=pd.DataFrame(revision_list)
    df.to_csv(to_csvfilename)
   
    return df
  
