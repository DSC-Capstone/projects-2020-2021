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
    readable[0].to_csv("data/raw/top1000.csv", index=False)
    return

