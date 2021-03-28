import sys
import scipy as sp
from scipy import signal
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, '../src')
from utils import *

# convert to Mbps
mbit_rate = 1 / 125000
 
# Data paths
two_four_fp = '../data/240p/'
three_six_fp = '../data/360p/'
four_eight_fp = '../data/480p/'
seven_two_fp = '../data/720p/'
ten_eight_fp = '../data/1080p/' 

# Read in data
def load_stdoan():
  stdoan_two_four = pd.read_csv(two_four_fp + "stdoan-101-action-240p-20201127.csv")
  stdoan_three_six = pd.read_csv(three_six_fp + "stdoan-101-action-360p-20201206.csv")
  stdoan_four_eight = pd.read_csv(four_eight_fp + "stdoan-101-action-480p-20201127.csv")
  stdoan_seven_two = pd.read_csv(seven_two_fp + "stdoan-101-action-720p-20201127.csv")
  stdoan_ten_eight = pd.read_csv(ten_eight_fp + "stdoan-101-action-1080p-20201127.csv")

  stdoan_two_four['resolution'] = '240p'
  stdoan_four_eight['resolution'] = '480p'
  stdoan_three_six['resolution'] = '360p'
  stdoan_seven_two['resolution'] = '720p'
  stdoan_ten_eight['resolution'] = '1080p'
  
  return [stdoan_two_four, stdoan_three_six, stdoan_four_eight, stdoan_seven_two, stdoan_ten_eight]

def load_iman():
  iman_two_four = pd.read_csv(two_four_fp + "imnemato-110-action-240p-20210202.csv")
  iman_three_six = pd.read_csv(three_six_fp + "imnemato-110-action-360p-20210213.csv")
  iman_four_eight = pd.read_csv(four_eight_fp + "imnemato-110-action-480p-20210202.csv")
  iman_seven_two = pd.read_csv(seven_two_fp + "imnemato-110-action-720p-20210213.csv")
  iman_ten_eight = pd.read_csv(ten_eight_fp + "imnemato-110-action-1080p-20210202.csv")

  iman_two_four['resolution'] = '240p'
  iman_four_eight['resolution'] = '480p'
  iman_three_six['resolution'] = '360p'
  iman_seven_two['resolution'] = '720p'
  iman_ten_eight['resolution'] = '1080p'

  return [iman_two_four, iman_three_six, iman_four_eight, iman_seven_two, iman_ten_eight]

# Comparing Byte Stream
def subplot_byte_stream(df1_lst, df2_lst, res_lst, byte_dir, xlim):

  color_lst = sns.color_palette()
  # preprocess data for plotting 
  x = np.arange(xlim)
  df1_bytes = [df.groupby('Time')[byte_dir].sum()  * mbit_rate for df in df1_lst]
  df2_bytes = [df.groupby('Time')[byte_dir].sum()  * mbit_rate for df in df2_lst]
  download_dict = {res_lst[i]: (df1_bytes[i], df2_bytes[i], color_lst[i]) for i in np.arange(len(res_lst))}

  # set up plot structure and labeling
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(5, 2, figsize=(24, 18), sharex=True, sharey=True)

  # plot line graphs
  plot_idx = 0
  for res in download_dict.keys():
    sns.lineplot(x, download_dict[res][0][:xlim], label=res, color=download_dict[res][2][:xlim], ax=axes[plot_idx, 0])
    sns.lineplot(x, download_dict[res][1][:xlim], label=res, color=download_dict[res][2][:xlim], ax=axes[plot_idx, 1])
    axes[plot_idx, 0].set_title("User 1 - " + res, fontsize=18)
    axes[plot_idx, 1].set_title("User 2 - " + res, fontsize=18)
    plot_idx += 1

  # aesthetic
  plt.suptitle('Download - All Resolutions (Action) - Youtube', fontsize=24)
  plt.subplots_adjust(top=0.93)
  plt.setp(axes, xlim=(0, xlim), xticks=np.linspace(0, xlim, num=(xlim/60) + 1, endpoint=True), ylim=(-1, 50), yticks=[0, 10, 20, 30, 40, 50])

  for ax in axes.flat:
      ax.set_xlabel("Seconds (from start)", fontsize=24)
      ax.set_ylabel("Mbps", fontsize=24)
      ax.label_outer()
  
  fig.show()

def rolling_bytes_stream(df_lst, res_lst, xlim, window_size_small, window_size_large, sample_size):
  color_lst = sns.color_palette()
  # preprocess data for plotting
  x_s = np.arange(xlim)
  x_l = np.arange(window_size_large, xlim + window_size_large)
  pre_rolling = [explode_extended(df).resample(sample_size, on='Time').sum()[['pkt_size']] for df in df_lst]
  rolling_s_sum = [df.rolling(window_size_small).mean().fillna(0) * mbit_rate for df in pre_rolling]
  rolling_l_sum = [df.rolling(window_size_large).mean().fillna(0) * mbit_rate for df in pre_rolling]
  rolling_sum_dict = {res_lst[i]: (rolling_s_sum[i], rolling_l_sum[i], color_lst[i]) for i in np.arange(len(res_lst))}
    
  # set up plot structure and labeling
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(5, 2, figsize=(24, 18), sharex=False, sharey=True)

  # plot line graphs
  plot_idx = 0
  for res in rolling_sum_dict.keys():
    sns.lineplot(
      x_s, rolling_sum_dict[res][0]['pkt_size'][:xlim],label=res, color=rolling_sum_dict[res][2], ax=axes[plot_idx, 0])
    
    sns.lineplot(
      x_l, rolling_sum_dict[res][1]['pkt_size'][window_size_large:xlim+window_size_large], label=res, color=rolling_sum_dict[res][2], ax=axes[plot_idx, 1])

    axes[plot_idx, 0].set_title("{}s Rolling Average - ".format(str(window_size_small)) + res, fontsize=18)
    axes[plot_idx, 1].set_title("{}s Rolling Average - ".format(str(window_size_large)) + res, fontsize=18)
    
    plot_idx += 1

  # aesthetic
  plt.suptitle('Moving Average - {}s v {}s'.format(str(window_size_small), str(window_size_large)), fontsize=24)
  plt.subplots_adjust(top=0.93)

  for ax in axes.flat:
      ax.set_xlabel("Seconds (from start)", fontsize=24)
      ax.set_ylabel("Mbps", fontsize=24)
      ax.label_outer()
  
  fig.show()

# Exploring Data Peaks
def preprocess_data_peaks(data_lst, byte_dir):
  peak_df = pd.DataFrame()

  # locate all peaks based on resolution
  for df in data_lst:
    temp_download_df = df[[byte_dir]].loc[get_peak_loc(df, byte_dir)] * mbit_rate
    temp_download_df.columns = ['Mbps']
    temp_download_df['Direction'] = 'Download' # was going to compare upload and download but the scale is too different
    temp_download_df['resolution'] = df['resolution'][0]
    peak_df = pd.concat((peak_df, temp_download_df))

  return peak_df

def subplot_peak_boxplot(peaks_df1, peaks_df2):

  # set up plot structure and labels
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

  # peaks_df generated by preprocess_data_peaks
  sns.boxplot(data=peaks_df1, x="resolution", y="Mbps", linewidth=2, ax=axes[0])
  axes[0].set_title("Peaks (All Resolutions) - User 1", fontsize=20)

  sns.boxplot(data=peaks_df2, x="resolution", y="Mbps", linewidth=2, ax=axes[1])
  axes[1].set_title("Peaks (All Resolutions) - User 2", fontsize=20)

  # aesthetic
  sns.despine(left=True)

  for ax in axes.flat:
      ax.set_xlabel("Resolutions", fontsize=14)
      ax.set_ylabel("Mbps", fontsize=14)
      ax.label_outer()
  
  fig.show()

def subplot_peak_kde_hist(peaks_df1, peaks_df2, res_lst):
  # set up plot structure and labels
  fig, axes = plt.subplots(2, 2, figsize=(16, 10), sharex=True)
  plt.setp(axes, xlim=(0, 25), xticks=[0, 5, 10, 15, 20, 25])

  # peaks_df generated by preprocess_data_peaks
  sns.histplot(data=peaks_df1, x="Mbps", hue="resolution", multiple="stack", legend=False, ax=axes[0,0])
  axes[0, 0].set_title("Peak Histogram - User 1", fontsize=16)
  sns.kdeplot(data=peaks_df1, x="Mbps", hue="resolution", legend=False, ax=axes[0,1])
  axes[0, 1].set_title("Peak Density - User 1", fontsize=16)

  sns.histplot(data=peaks_df2, x="Mbps", hue="resolution", multiple="stack", legend=False, ax=axes[1,0])
  axes[1, 0].set_title("Peak Histogram - User 2", fontsize=16)
  sns.kdeplot(data=peaks_df2, x="Mbps", hue="resolution", legend=False, ax=axes[1,1])
  axes[1, 1].set_title("Peak Density - User 2", fontsize=16)

  # aesthetic
  fig.legend(
    res_lst[::-1],
    loc="lower center",
    title="Resolution",
    ncol=len(res_lst)
  )

  fig.subplots_adjust(bottom=0.1)

  plt.suptitle('Peak Distributions', fontsize=20)
  plt.subplots_adjust(top=0.90)
  
  fig.show()

# Spectral Analysis
def subplot_periodogram(df1, df2, size, res_lst):
  sns.set_style('whitegrid')
  fig, axes = plt.subplots(5, 2, figsize=(24, 18), sharex=True, sharey=False)

  color_lst = sns.color_palette()

  df1_resample_lst = [explode_extended(df).resample(size, on='Time').sum() * mbit_rate for df in df1]
  df2_resample_lst = [explode_extended(df).resample(size, on='Time').sum() * mbit_rate for df in df2]

  # plot user 1
  i = 0
  for df in df1_resample_lst:
    f, Pxx = sp.signal.welch(df['pkt_size'], fs=.5)
    sns.lineplot(f, Pxx, ax=axes[i, 0], label=res_lst[i], color=color_lst[i])
    axes[i, 0].set_title("User 1 - {}".format(res_lst[i]), fontsize=16)
    i += 1

  # plot user 2
  j = 0
  for df in df2_resample_lst:
    f, Pxx = sp.signal.welch(df['pkt_size'], fs=.5)
    sns.lineplot(f, Pxx, ax=axes[j, 1], label=res_lst[j], color=color_lst[j])
    axes[j, 1].set_title("User 2 - {}".format(res_lst[j]), fontsize=16)
    j += 1

  # aesthetic

  plt.suptitle('Periodogram - 2s Resample (Binning)', fontsize=20)
  plt.subplots_adjust(top=0.90)
    
  fig.show()