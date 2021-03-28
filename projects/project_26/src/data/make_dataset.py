import pandas as pd
import numpy as np
import os
from bs4 import BeautifulSoup
import twint
from pytrends.request import TrendReq
import pageviewapi

##### For Wikipedia #####

def get_xml(titles):
    '''
    Scrapes the xml for given list of Wikipedia pages

    :param titles: list of wiki article names
    '''
    url_part1 = 'https://en.wikipedia.org/w/index.php?title=Special:Export&pages='
    url_part2 = '&history=1&action=submit'
    
    for title in titles:
        url = url_part1 + title + url_part2
        response = requests.get(url)   #web request

    #create wiki folder
    if not os.path.isdir("data/raw/wiki"):
        os.mkdir("../data/raw/wiki")
        
    with open('data/raw/wiki'+ title +'_wiki.xml', 'wb') as file:
        file.write(response.content)
        
    response.close()
    
def xml_to_soup(fp):
    '''
    Processes xml data using beautiful soup and
    returns list of data for each page

    :param fp: xml input filepath
    :return: list of data
    '''
    content = []
    with open(fp, encoding = 'utf8') as file:

        content = file.readlines()
        content = "".join(content)
        soup = BeautifulSoup(content, "xml")
        
    pages = soup.findAll("page")
    return pages[0]

def soup_to_df(page):
    '''
    Converts soupified xml data for Wiki pages 
    into dataframe
    
    :param page: list of xml data for each page
    :return: xml data as dataframe
    '''
    data = []

    title = page.title.text
    revisions = page.findAll("revision")

    for revision in revisions:
        r_id = revision.id.text 
        time = revision.timestamp.text
        try:
            try:
                username = revision.contributor.username.text
            except: 
                username = revision.contributor.ip.text
        except:
            username = 'N/A'
        text = revision.format.next_sibling.next_sibling.text
        data.append([title, r_id, time, username, text])

    df = pd.DataFrame(data, columns = ['title', 'id', 'time', 'username', 'text'])

    hist = [] #history of text
    version = [] #edit version
    username = []
    revert = [] #0 or 1
    curr = 1 #to keep track of version
    length = [] #length of text

    for idx, row in df.iterrows():
        if row.text not in hist: # not a revert
            hist.append(row.text)
            version.append(curr)
            username.append(row.username)
            length.append(len(row.text))
            revert.append('0')
            curr += 1
        else: #is revert
            temp = hist.index(row.text)
            version.append(version[temp])
            username.append(row.username)
            length.append(len(row.text))

            #if self revert
            if row.username == username[version[temp]]:
                revert.append('0')
            else:
                revert.append('1')

    df['version'] = version
    df['revert'] = revert
    df['length'] = length
    return df


def df_to_ld(df, outpath):
    '''
    Given a list of cleaned dataframes from xml data,
    produces light dump file into data/raw

    :param df: cleaned xml dataframes
    :param outpath: output filepath for lightdumps    
    '''
    light_dump = ''
    
    title = df.title[0]
    light_dump = light_dump + title + '\n'
    for idx, row in df.iterrows():
        line = '^^^_' + row.time + ' ' + row.revert + ' ' + str(row.version) + ' ' + str(row.length) + ' ' + row.username 
        light_dump = light_dump + line + '\n'
    
    with open(outpath, 'w') as f:
        f.write(light_dump)
    repo = 'XML Converted to light dump at ' + outpath
    print(repo)

def xml_to_light_dump(fp, outfp):
    '''
    Given an input file path and output path, 
    turns the xml file into a light dump 
    and stores it at the output file path

    :param fp: input xml filepath
    :param outfp: output filepath for lightdump
    :return: lightdump
    '''
    # create light dump directory first
    if not os.path.isdir("data/raw/wiki/light_dump"):
        os.mkdir("../data/raw/wiki/light_dump")
    
    # convert to light dump
    soup = xml_to_soup(fp)
    df = soup_to_df(soup)
    return df_to_ld(df, outfp)


def read_lightdump(fp):
    '''
	Reads in n lightdump pages and returns a list of all titles 
    read and their corresponding data as a DataFrame
	:param fp: input filepath
	:param n: number of articles to read
	:return: list of article titles, list of corresponding article lightdump data as DataFrame
	'''
    with open(fp) as file:
        df = pd.DataFrame(columns = ['date', 'revert', 'revision_id', 'length', 'user'])
        for line in file:
            if '^^^_' not in line:
                title = line.strip('\n').strip()

            else:
                data = line.strip("^^^_").strip('\n').split()
                row = pd.Series(dtype = 'object')

                row['date'] = data[0]
                row['revert'] = int(data[1])
                row['revision_id'] = int(data[2])
                row['length'] = int(data[3])
                row['user'] = data[4]

                df = df.append(row, ignore_index = True)

    df['date'] = pd.DatetimeIndex(pd.to_datetime(df.date)).tz_localize(None)
    df.date = df.date.apply(lambda x: x.date())
    return title, df


def get_data(wiki_fp):
    
    wiki_ld = os.listdir(wiki_fp)
    wiki_ld.sort()
    files = []

    for dump in wiki_ld:
        fp = os.path.join(wiki_fp, dump)
        files.append(fp)
            
    data = []
    for fp in files:
        try:
            title, df = read_lightdump(fp)
        except:
            print("Cannot read file: " + fp)
            continue
        data.append((title, df))
    return data
    


##### For Twitter #####

def tweets_query(search, since, until, pandas=True, csv=False, output='out.csv'):
    '''
    Query tweets with search terms between given dates
    Sample query: tweets_query('#whitelivesmatter', '2020–05–20', '2020–05–30')

    :param search: search query
    :param since: query start date
    :param until: query end date
    :param pandas: whether or not to return pandas df, default True
    :param csv: whether or not to save as csv, default False
    :param output: output filename, default 'out.csv'
    :return: tweets dataframe
    '''
    c = twint.Config()
    c.Search = search
    c.Since = since
    c.Until = until
    c.Hide_output = True
    c.Pandas=True
    c.Store_object = True
    c.Count = True
    c.Store_csv = csv
    c.Output = output
    twint.run.Search(c)
    Tweets_df = twint.storage.panda.Tweets_df
    return Tweets_df

def normalize_dates(df, release_date, start=-2, end=10):
    '''
    Normalizes dates to release date, only keeping "start" days
    before to "end" days after. Returns a copy.

    :param df: input album dataframe
    :param release_date: album release date
    :param start: number of days before release date to keep
    :param end: number of days after release date to keep
    :return: album dataframe with normalized dates
    '''
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    date_diff = (pd.Timestamp(release_date) - df['date'].min())
    normalized_dates = (pd.factorize(df['date'], sort=True)[0] - date_diff.days).astype(object)
    
    # remove out of scope
    normalized_dates[(normalized_dates > end) | (normalized_dates < start)] = np.NaN
    df['normalized_dates'] = normalized_dates
    return df.dropna(subset=['normalized_dates'])


##### For Google Trends #####
    
def query_trends(term_list,
                 category = None, dates = 'all'):
    '''
    Query Google Trends data with search terms between
    given dates
    
    :param term_list: list of terms to search
    :param category: optional Google Trends category
    :param dates: start and end dates of query search in 'yyyy-mm-dd yyyy-mm-dd' format
    
    :return: DataFrame with Google Trends popularity for the search terms in the given timeframe
    '''
    pytrends = TrendReq()
    
    pytrends.build_payload(kw_list = term_list,
                           timeframe = dates)
    
    df = pytrends.interest_over_time()
    
    df.drop('isPartial', inplace = True, axis = 1)
    df.reset_index(inplace = True)
    
    df = df.melt(id_vars = ['date'],
                var_name = 'artist',
                value_name = 'popularity')
    
    return df


##### For Wikipedia Page Views #####

def query_per_article(page, start, end, interval = 'daily'):
    '''
    Query Wikipedia page view data from English Wikipedia
    for a specific page between given dates
    
    :param page: specific Wikipedia page to query
    :param start: start date in yyyy-mm-dd format, cannot be prior to 2015-07-01
    :param end: end date in yyyy-mm-dd format, cannot be prior to 2015-07-01
    :param interval: date interval such as daily, monthly, etc.
    
    :return: DataFrame listing page views for a single article within a specific timeframe
    '''
    raw_page = pageviewapi.per_article('en.wikipedia', page, start,
                                        end, granularity = interval)
    
    page_views = raw_page.items()
    page_views = list(page_views)[0][1]
    page_views = pd.DataFrame.from_dict(page_views)
    page_views = page_views[['article', 'timestamp', 'views']]

    page_views['timestamp'] = page_views['timestamp']\
        .apply(lambda x: x[:4] + '-' + x[4:6] + '-' + x[6:8])

    page_views['article'] = page_views['article']\
        .apply(lambda y: y.replace('_', ' '))
    
    return page_views