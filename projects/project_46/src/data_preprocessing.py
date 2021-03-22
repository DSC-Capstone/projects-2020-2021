import pandas as pd
import numpy as np
import datetime
from os import listdir
from bs4 import BeautifulSoup
from tqdm import tqdm
import re
import json
import seaborn as sns


# Part 1 - Clean 8K reports
def clean_doc_text(doc_text):
    doc_text = re.sub('\n+', '\n', doc_text)
    cleaned_text = ''
    for sent in doc_text.split('\n'):
        cleaned_text += re.sub('\s+', ' ', sent)
        cleaned_text += '\n'
    doc_text = re.sub('\\t', ' ', doc_text)
    return doc_text

def handle_single_document(doc, all_failed_doc):
    time, event_type = None, None
    if doc.strip() == '\n' or doc == '\n' or doc == '':
        pass
    elif 'EVENTS:' not in doc or 'TIME:' not in doc:
        all_failed_doc.append(doc)
    else:
        time = re.findall('TIME:.+\\n', doc)[0]
        event_type = re.findall('EVENTS:.+\\n', doc)[0]
    return time, event_type

def handler_clean_8k(data_dir):
    # print('===================================================================')
    print(' => Cleaning 8K...')
    # print('===================================================================')
    all_data_dict = {}
    counter_8k = 0
    raw_8k_fp = data_dir + 'raw/8K-gz/'
    # raw_8k_fp = data_dir + ''
    # print('raw_8k_fp', raw_8k_fp)
    # print('data_dir', data_dir)
    for fp in tqdm(listdir(raw_8k_fp)):
        if fp == '.DS_Store':
            continue
        full_path = raw_8k_fp + fp
        file = open(full_path, 'r')
        tmp_txt = ''
        for line in file:
            tmp_txt += line

        all_failed_doc = []
        tmp_all_docs = []
        for doc in tmp_txt.split('</DOCUMENT>'):
            doc = doc.replace('</DOCUMENT>', '')
            time, event_type = handle_single_document(doc, all_failed_doc)
            if time:
                tmp_dict = {}
                tmp_dict['time'] = time
                tmp_dict['event_type'] = event_type
                tmp_dict['full_text'] = clean_doc_text(doc)
                tmp_all_docs.append(tmp_dict)
        all_data_dict[fp] = tmp_all_docs

        counter_8k += 1
        # if counter_8k > 50:
        #     break

    print(' => Saving the cleaned 8K report to local dir: data_dir/processed/8k.json')
    with open(data_dir + 'processed/8k.json', 'w') as outfile:
        json.dump(all_data_dict, outfile)
    print(' => Example of one of the cleaned 8k report:', all_data_dict['AAPL'][0])
    print()
    return all_data_dict

# Part 2 - Process EPS
def get_EPS(file):
    result = []
    soup = BeautifulSoup(open(file), "html.parser")
    lst = soup.find_all('small')
    for tag in lst:
        a = tag.find('a')
        if a != None and a.text.isupper():
            temp = [a.text]
            p = a.parent
            for i in range(3):
                p = p.findNext('small')
                temp.append(p.text)
            result.append(temp)
    return result
def handler_process_eps(data_dir):
    # print('===================================================================')
    print(' => Processing EPS...')
    # print('===================================================================')
    result = []
    raw_eps_fp = data_dir + 'raw/EPS/'
    for file in tqdm(listdir(raw_eps_fp)):
        # print(file)
        if 'txt' not in file:
            continue
        temp = get_EPS(raw_eps_fp + file)
        date = int(file.split('.')[0])
        for t in temp:
            result.append([date] + t)

    df = pd.DataFrame(result, columns = ['Report Date', 'Code', 'Surprise(%)', 'Reported EPS', 'Consensus EPS'])
    print(' => Saving the processed EPS infomation to local dir: [data_dir]/processed/EPS.csv')
    print()
    df.to_csv(data_dir + 'processed/EPS.csv', index = None)

# Part 3 - Merge EPS and 8K text
def clean_time(time_str):
    time_str = re.sub('[^0-9]', '', time_str)
    date = time_str[:8]
    time = time_str[8:]
    return date, time

def merge_EPS_8K(data_dir, from_local_file = False):
    if from_local_file:
        with open(data_dir + 'processed/8k.json') as json_file:
            all_data_dict = json.load(json_file)
    else:
        all_data_dict = handler_clean_8k(data_dir) # output of step 1

    all_8k_lst = []
    doc_counter = 0
    for symbol in tqdm(all_data_dict.keys()):
        tmp_docs = all_data_dict[symbol]
        for doc in tmp_docs:
            doc_counter += 1
            date, time = clean_time(doc['time'])
            all_8k_lst.append({
                'date': date,
                'time': time,
                'event_type': doc['event_type'],
                'full_text': doc['full_text'],
                'Code': symbol
            })
        # if doc_counter > 50:
        #     break
    all_8k_df = pd.DataFrame(all_8k_lst)
    all_8k_df['time_code'] = all_8k_df.date + all_8k_df.Code

    eps_df = pd.read_csv(data_dir + 'processed/EPS.csv') # output of step 2
    eps_df['time_code'] = eps_df['Report Date'].apply(lambda x: str(x)) + eps_df.Code
    # print(all_8k_df.head(1))
    # print(eps_df.head(1))
    merged_df = all_8k_df.merge(eps_df, how = 'inner', on = 'time_code').dropna()
    merged_df = merged_df.drop(columns = ['Code_x', 'Report Date', 'time_code'])\
        .rename(columns = {'Code_y': 'symbol'})
    merged_df['hr'] = merged_df.time.apply(lambda time: float(time[:4]) / 100)
    merged_df['pre_market'] = merged_df['hr'] < 9.3
    merged_df.date = merged_df.date\
        .apply(lambda x: pd.to_datetime(x[:4] + '-' + x[4:6] + '-' + x[6:]))
    return merged_df

# Part 4
def calc_price_changes(price_df, date_delta, most_recent_dates):
    price_delta = []
    # print(price_df.date_idx, most_recent_dates)
    # print(price_df.head())
    for date_idx in most_recent_dates:
        if date_idx == -1:
            price_delta.append(np.nan)
            continue

        # Check if previous trading date is logged
        prev_date = date_idx - date_delta
        while prev_date not in price_df.date_idx.values and prev_date >= 0:
            prev_date -= 1
        if prev_date < 0:
            price_delta.append(np.nan)
            continue

        # Find prev info
        # print(price_df.query('date_idx == ' + str(prev_date)))
        prev_close = price_df.query('date_idx == ' + str(prev_date))['Adj Close'].values[0]
        curr_close = price_df.query('date_idx == ' + str(date_idx))['Adj Close'].values[0]
        percent_change = round((curr_close - prev_close) / prev_close * 100, 2)
        price_delta.append(percent_change)
    # print(price_delta)
    # print(sum([np.isnan(i) for i in price_delta]))
    return price_delta

def calc_prediction_target_vix(price_df, dates_pairs, sp500_dict, vix_dict):
    price_delta = []
    vix_values = []
    for pair in dates_pairs:
        curr_date, target_date = pair
        if curr_date == -1 or target_date == -1:
            price_delta.append(np.nan)
            vix_values.append(np.nan)
            continue

        # Find stock price change info
        curr_close = price_df.query('date_idx == ' + str(curr_date))['Adj Close'].values[0]
        target_close = price_df.query('date_idx == ' + str(target_date))['Adj Close'].values[0]
        stock_percent_change = round((target_close - curr_close) / curr_close * 100, 2)

        # Fine sp500 price change info
        sp500_change = sp500_dict[target_date]
        price_delta.append(stock_percent_change - sp500_change)

        # Find the last vix close value
        last_vix = vix_dict[curr_date]
        vix_values.append(last_vix)
    return price_delta, vix_values

def prep_sp500(min_date, data_dir):
    # print(min_date)
    sp500_df = pd.read_csv(data_dir + 'raw/sp500.csv')
    sp500_df.Date = sp500_df.Date.apply(lambda x: pd.to_datetime(x))
    sp500_df['date_idx'] = sp500_df.Date.apply(lambda x: (x - min_date).days)
    sp500_df = sp500_df.query('date_idx >= 0').reset_index(drop = True)
    sp500_df.index = sp500_df.date_idx
    sp500_dict = dict(sp500_df.day_change)
    return sp500_dict

def prep_vix(min_date, data_dir):
    vix_df = pd.read_csv(data_dir + 'raw/price_history/vix.csv')
    vix_df = vix_df[['Date', 'Adj Close']].rename(columns = {'Adj Close': 'Close'})
    vix_df.Date = vix_df.Date.apply(lambda x: pd.to_datetime(x))
    vix_df['date_idx'] = vix_df.Date.apply(lambda x: (x - min_date).days)
    vix_df = vix_df.query('date_idx >= 0').reset_index(drop = True)
    vix_df.index = vix_df.date_idx
    vix_dict = dict(vix_df['Close'])
    return vix_dict

def handle_calc_dataset(df):
    cutoffs = np.percentile(df.date_idx, 50), np.percentile(df.date_idx, 75)
    def calc_dataset(x):
        if x <= cutoffs[0]:
            return 'train'
        elif x <= cutoffs[1]:
            return 'val'
        return 'test'
    return df.date_idx.apply(lambda x: calc_dataset(x))

def handle_merge_eps8k_pricehist(data_dir):
    print()
    print('===================================================================')
    print(' => Merging eps, 8k and price history...')
    # print('===================================================================')
    merged_df = merge_EPS_8K(data_dir) # Call part 3 code
    # print(merged_df)
    print(' => Done merging 8k and EPS!')
    min_date = merged_df.date.min()
    sp500_dict = prep_sp500(min_date, data_dir)
    vix_dict = prep_vix(min_date, data_dir)

    merged_df['date_idx'] = merged_df.date.apply(lambda x: (x - min_date).days)
    max_date_idx = merged_df['date_idx'].max()

    symbol_missing_price = []
    sub_dfs = []
    # print(merged_df.shape)
    price_history_dir = data_dir + 'raw/price_history/'
    for symbol in tqdm(merged_df.symbol.unique()):
        try:
            price_hist_df = pd.read_csv(price_history_dir + symbol + '.csv')
        except:
            symbol_missing_price.append(symbol)
            continue

        # pre-process price_history
        price_hist_df.Date = price_hist_df.Date.apply(lambda x: (pd.to_datetime(x)))
        price_hist_df['date_idx'] = price_hist_df.Date.apply(lambda x: (x - min_date).days)

        # Get intended dates
        tmp_merged_df = merged_df.query('symbol == "' + symbol + '"').reset_index(drop = True)

        most_recent_dates = []
        pred_rarget_dates = []
        for index, row in tmp_merged_df.iterrows():
            most_recent_date_idx = row.date_idx
            target_date_idx = row.date_idx
            if row.pre_market:
                most_recent_date_idx -= 1
            else:
                target_date_idx += 1

            # Adjust most recent date
            while most_recent_date_idx not in price_hist_df.date_idx.values and most_recent_date_idx >= 0:
                most_recent_date_idx -= 1
            if most_recent_date_idx <= 0:
                most_recent_date_idx = -1

            # Adjust target date
            while target_date_idx not in price_hist_df.date_idx.values and target_date_idx <= max_date_idx:
                target_date_idx += 1
            if target_date_idx > max_date_idx:
                target_date_idx = -1

            pred_rarget_dates.append((most_recent_date_idx, target_date_idx))
            most_recent_dates.append(most_recent_date_idx)

        # find out price changes
        for date_delta in [7, 30, 90, 365]:
            tmp_price_changes = calc_price_changes(price_hist_df, date_delta, most_recent_dates)
            tmp_merged_df['price_change_' + str(date_delta)] = tmp_price_changes
        targe_price_change, prev_vix_values = \
            calc_prediction_target_vix(price_hist_df, pred_rarget_dates, sp500_dict, vix_dict)
        tmp_merged_df['targe_price_change'] = targe_price_change
        tmp_merged_df['prev_vix_values'] = prev_vix_values
        sub_dfs.append(tmp_merged_df)
    updated_merged_df = pd.concat(sub_dfs).dropna().reset_index(drop = True)
    updated_merged_df['dataset'] = handle_calc_dataset(updated_merged_df)
    return updated_merged_df
