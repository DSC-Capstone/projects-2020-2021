import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.fftpack import fft, rfft,rfftfreq
from statsmodels.tsa.seasonal import seasonal_decompose

np.warnings.filterwarnings('ignore')

def get_date_list(start,end):
    start = datetime.datetime.strptime(start, "%Y-%m-%d")
    end = datetime.datetime.strptime(end, "%Y-%m-%d")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0, (end-start).days)]
    date_list = []
    for date in date_generated:
        date_list.append(date.strftime("%Y-%m-%d"))
    return date_list

def plot_score_cases(detrended, score_list, date_list, topath):
    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(16, 6))

    ax = sns.lineplot(y = score_list, x = date_list, label = 'Sentiment Score')
    ax2 = sns.lineplot(y = detrended/1500000+np.mean(score_list), x = date_list, label = 'Detrended Daily Cases')
    plt.xticks(rotation=30)
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    ax.legend()
    plt.xlabel('Date')
    plt.ylabel('Standardized unit')
    plt.title('Daily Sentiment Score vs. Detrended Daily Cases')
    fig.savefig(topath)
    return

def half_life(ts):  
    ts = np.asarray(ts)
    delta_ts = np.diff(ts)
    lag_ts = np.vstack([ts[1:], np.ones(len(ts[1:]))]).T
    beta = np.linalg.lstsq(lag_ts, delta_ts)
    return (np.log(2) / beta[0])[0]

def fourier(data,title,topath):
    nobs = len(data)
    x_ft = np.abs(rfft(data))
    x_freq = rfftfreq(nobs)
    fig = plt.figure(figsize=(10, 7))
    plt.plot(x_freq[2:], x_ft[2: ])
    plt.title(title)
    plt.xlabel('frequency (1/day)')
    plt.ylabel('magnitude')
    fig.savefig(topath)
    return

def detrend_data(daily_cases):
    daily_cases = pd.read_csv(daily_cases)
    daily_cases.date = pd.to_datetime(daily_cases.date)
    daily_cases = daily_cases.set_index('date')
    result_mul = seasonal_decompose(daily_cases['new_cases'], model='multiplicative', extrapolate_trend='freq')
    detrended = daily_cases.new_cases - result_mul.trend
    return detrended

def make_prediction(**kwargs):
    datapath, scorepath, casecsv, scorecsv, dates = kwargs['data_path'], kwargs['score_path'], kwargs['case_csv'], kwargs['score_csv'], kwargs['dates']
    if kwargs['test']:
        dates = kwargs['dates_test']

    date_list = get_date_list(dates[0], dates[1])
    detrended = detrend_data(datapath+casecsv)
    score_list = list(pd.read_csv(scorepath+scorecsv)['score'])
    plot_score_cases(detrended,score_list,date_list,scorepath+kwargs['score_plot'])
    corr, _ = pearsonr(detrended, score_list)

    print('Pearsons correlation: %.3f' % corr)
    print('Mean aversion time for Daily cases : '+ str(half_life(detrended)))
    print('Mean aversion time for Sentiment score : '+str(half_life(score_list)))
    fourier(score_list, 'Decomposition of Sentiment Score',scorepath+kwargs['fourier_plots'][0])
    fourier(list(detrended), 'Decomposition of Daily Cases',scorepath+kwargs['fourier_plots'][1])

    return