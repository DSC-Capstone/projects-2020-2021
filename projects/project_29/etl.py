import os
import sys
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup

def top_1000():
    '''
    get top 1000 articles of COVID in wikipedia from 2020/1/1 to 2020/12/31
    '''
    url="https://en.wikipedia.org/wiki/Wikipedia:WikiProject_COVID-19/Popular_pages"
    response=requests.get(url)
    page=BeautifulSoup(response.text, 'html.parser')
    table=page.find_all('table')[3]
    readable=pd.read_html(table.prettify())
    if not os.path.exists('data/raw'):
        os.makedirs('data/raw')
    readable[0].to_csv("data/raw/top1000.csv", index=False)
    return

def api_getpageview(article):
    '''
    Using GET method to get pageview data since 2020010100 (2020/1/1) to 2020123100 (2020/12/31)

    article: The article we are looking for

    return: A csv contained information
    '''
    date=[]
    pageview=[]
    result={}
    article=article.replace(" ", "_")
    head={}
    head["user-agent"]="Googlebot/2.1 (+http://www.google.com/bot.html)"
    url="https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/{0}/daily/2020010100/2020123100".format(article)
    response=requests.get(url, headers=head)
    try:
        js=response.json()['items']
        for i in js:
            pageview.append(int(i['views']))
            date.append((i['timestamp']))
        result['timestamp']=date
        result['pageview']=pageview
        result=pd.DataFrame(result)

        return result
    except Exception as e:
        print(article)
        print("this article is too new to get data")
        return
   

def pageview_csv(data, outpath):
    '''
    read data and generate csv to data/pageview

    data: road to data
    outpath: output path to data

    return: null, but output csv to data/pageview
    '''
    source=pd.read_csv(data)['Page title'].values
    arranged=[]
    for i in source:
        i=i.replace(" ", "_")
        i=i.replace("/", "%2F")
        arranged.append(i)
    counter=0
    for j in arranged:
        csv_path=j+"_pageview.csv"
        out_path=outpath
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file=os.path.join(out_path,csv_path)
        result=api_getpageview(j)
        try:
            result.to_csv(out_file, index=False)
        except Exception as e:
            counter=counter
        counter +=1
        if counter%100==0:
            print("number of articles' pageviews made", counter)
    return

def get_revisions(title,outpath):
    S = requests.Session()

    URL = "https://en.wikipedia.org/w/api.php"
    sttime="2020-12-31T00:00:00Z"
    counter=0
    judge=True
    output=outpath
    if not os.path.exists(output):
        os.makedirs(output)
    while judge:

        PARAMS = {
            "action": "query",
            "prop": "revisions",
            "rvprop": "timestamp|user|comment",
            "rvslots": "main",
            "rvend":"2020-01-01T00:00:00Z",
            "rvlimit":"500",
            "formatversion": "2",
            "format": "json"
        }
        PARAMS['titles']=title
        PARAMS['rvstart']=sttime

        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        history=DATA['query']['pages'][0]['revisions']
        df=pd.DataFrame(history)
        outfile=title+'_revision_'+str(counter)+'.csv'
        outroute=os.path.join(output, outfile)
        df.to_csv(outroute, index=False)
        if len(df)<500:
            judge=False
        else:
            counter +=1
            sttime=df['timestamp'].values[-1]
    return 