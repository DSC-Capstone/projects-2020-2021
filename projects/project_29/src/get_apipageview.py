import pandas as pd
import json
import requests
import os

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
    js=response.json()['items']
    for i in js:
        pageview.append(int(i['views']))
        date.append((i['timestamp']))
    result['timestamp']=date
    result['pageview']=pageview
    result=pd.DataFrame(result)

    return result


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
        result.to_csv(out_file, index=False)
        counter +=1
        if counter%100==0:
            print("number of articles' pageviews made", counter)
    return