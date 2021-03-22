import bs4 as bs 
import pickle
import requests
import pandas as pd
import re

'''
This method is to scrape the yahoo finance website for all the dowjones tickers

Returns: The Dow Jones tickers
'''
def save_dow_tickers():

    tickers = 'WMT WBA VZ V UNH TRV PG NKE MSFT MRK MMM MCD KO JPM JNJ INTC IBM HON HD GS DOW DIS CVX CSCO CRM CAT BA AXP AMGN AAPL'
    return tickers.split(' ');

'''
This method webscrapes from wikipedia to get all the SP500 tickers 

Returns: Sp500 tickers

'''
def save_sp500_tickers():
    data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    table = data[0]
    #sliced_table = table[1:]
    #header = table.iloc[0]
    #corrected_table = sliced_table.rename(columns=header)
    tickers = list(table[1:]['Symbol'])
    return tickers


