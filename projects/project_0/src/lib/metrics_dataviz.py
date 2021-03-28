import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def sci_largest_ratio(politicians, inpath, outpath):
    """
    takes in list of names for politicians and creates line
    graph for their likes over months
    * takes politician with highest tweet ratio average and 
    lowest tweet ratio average from scientific group
    """
    with open(inpath) as f:
        data = json.load(f)

    plt.figure(figsize=(12, 5))
    
    p1 = politicians[0]
    x1 = list(data[p1].keys())[100:-7]
    y1 = list(data[p1].values())[100:-7]
    plt.plot(x1, y1, color = 'blue')
    
    p2 = politicians[1]
    x2 = list(data[p2].keys())[100:-7]
    y2 = list(data[p2].values())[100:-7]
    plt.plot(x2, y2, color = 'dodgerblue')
        
    plt.suptitle('Highest Ratio Politician vs Lowest Ratio Politician for Scientific Group')
    plt.title('Rolling average of Tweet likes with 4 month window')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Number of Likes')
    plt.legend(politicians)

    plt.savefig(outpath + '/sci_ratio_comparison.png', bbox_inches='tight')
    
def misinfo_largest_ratio(politicians, inpath, outpath):
    """
    takes in list of names for politicians and creates line
    graph for their likes over months
    * takes politician with highest tweet ratio average and 
    lowest tweet ratio average from misinfo group
    """
    with open(inpath) as f:
        data = json.load(f)
    
    plt.figure(figsize=(12, 5))

    p1 = politicians[0]
    x1 = list(data[p1].keys())[100:-7]
    y1 = list(data[p1].values())[100:-7]
    plt.plot(x1, y1, color = 'orange')
    
    p2 = politicians[1]
    x2 = list(data[p2].keys())[100:]
    y2 = list(data[p2].values())[100:]
    plt.plot(x2, y2, color = 'orangered')
        
    plt.suptitle('Highest Ratio Politician vs Lowest Ratio Politician for Misinformation Group')
    plt.title('Rolling average of Tweet likes with 4 month window')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Number of Likes')
    plt.legend(politicians)
        
    plt.savefig(outpath + '/misinfo_ratio_comparison.png', bbox_inches='tight')
    
def both_largest_ratio(politicians, sci_inpath, misinfo_inpath, outpath):
    """
    takes in list of names for politicians and creates line
    graph for their likes over months
    * takes politicians with highest tweet ratio average from both groups 
    """
    with open(sci_inpath) as f:
        sci_data = json.load(f)
    
    plt.figure(figsize=(12, 5))
    p1 = politicians[0]
    x1 = list(sci_data[p1].keys())[100:-7]
    y1 = list(sci_data[p1].values())[100:-7]
    plt.plot(x1, y1, color = 'blue')
    
    with open(misinfo_inpath) as f:
        misinfo_data = json.load(f)
    
    p2 = politicians[1]
    x2 = list(misinfo_data[p2].keys())[100:-7]
    y2 = list(misinfo_data[p2].values())[100:-7]
    plt.plot(x2, y2, color = 'orange')
        
    plt.suptitle('Highest Ratio Politicians from Scientific and Misinformation Groups')
    plt.title('Rolling average of Tweet likes with 4 month window')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Number of Likes')
    plt.legend(politicians)
        
    plt.savefig(outpath + '/both_ratio_comparison.png', bbox_inches='tight')


def most_likes_comparison(politicians, sci_inpath, misinfo_inpath, outpath):
    """
    takes in the politicians with the most likes for both groups
    compares the likes over time 
    """
    with open(sci_inpath) as f:
        sci_data = json.load(f)
        
    plt.figure(figsize=(12, 5))
    
    p1 = politicians[0]
    x1 = list(sci_data[p1].keys())[100:-7]
    y1 = list(sci_data[p1].values())[100:-7]
    plt.plot(x1, y1, color = 'blue')
    
    with open(misinfo_inpath) as f:
        misinfo_data = json.load(f)
    
    p2 = politicians[1]
    x2 = list(misinfo_data[p2].keys())[100:-7]
    y2 = list(misinfo_data[p2].values())[100:-7]
    plt.plot(x2, y2, color = 'orangered')
    
    plt.suptitle('Highest Number of Likes from Scientitic and Misinformation Groups')
    plt.title('Rolling average of Tweet likes with 4 month window')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Average Number of Likes over Time')
    plt.legend(politicians)
 
    plt.savefig(outpath + '/both_most_likes_comparison.png', bbox_inches='tight')

def most_tweets_comparison(politicians, sci_inpath, misinfo_inpath, outpath):
    """
    takes in the politicians with the most tweets for both groups
    compares the likes over time 
    """
    with open(sci_inpath) as f:
        sci_data = json.load(f)
        
    plt.figure(figsize=(12, 5))
    
    p1 = politicians[0]
    x1 = list(sci_data[p1].keys())
    y1 = list(sci_data[p1].values())
    plt.plot(x1, y1, color = 'blue')
    
    with open(misinfo_inpath) as f:
        misinfo_data = json.load(f)
    
    p2 = politicians[1]
    x2 = list(misinfo_data[p2].keys())
    y2 = list(misinfo_data[p2].values())
    plt.plot(x2, y2, color = 'orangered')
    
    plt.suptitle('Highest Number of User Tweets from Scientitic and Misinformation Groups')
    plt.title('Rolling average of Tweet likes over last 200 Tweets')
    plt.xlabel('Number of Tweets')
    plt.xticks(rotation=90) 
    plt.xticks(range(0,4001,1000))
    plt.ylabel('Average Number of Likes')
    plt.legend(politicians)
 
    plt.savefig(outpath + '/both_most_tweets_comparison.png', bbox_inches='tight')

def group_sum_over_year(sci_inpath, misinfo_inpath, outpath):
    '''
    makes chart to show how full group's likes have changed over years 
    '''
    with open(sci_inpath) as f:
        sci_data = json.load(f)
        
    with open(misinfo_inpath) as f:
        mis_data = json.load(f)
        
    fig, ax = plt.subplots(2,1, figsize=(10, 8), sharey = True, sharex=True)
    
    x1 = list(sci_data.keys())[4:]
    y1 = []
    for year in x1:
        y1.append(sum(sci_data[year]))
    ax[0].bar(x1, y1, color='blue')
    ax[0].legend(['Scientific'])
    
    x2 = list(mis_data.keys())[4:]
    y2 = []
    for year in x2:
        y2.append(sum(mis_data[year]))
    ax[1].bar(x2, y2, color='orangered')
    ax[1].legend(['Misinformation'], loc='upper left')
    
    ax[0].set_title('Total Number of Likes per year for both groups')
    plt.xlabel('Year')
    plt.ylabel('Total Number of Likes')
    
    plt.savefig(outpath + '/both_group_sum_over_year.png', bbox_inches='tight')

def normalized_group_sum_over_year(sci_inpath, misinfo_inpath, outpath):
    '''
    makes chart to show how full group's likes have changed over years 
    '''
    with open(sci_inpath) as f:
        sci_data = json.load(f)
        
    with open(misinfo_inpath) as f:
        mis_data = json.load(f)
        
    fig, ax = plt.subplots(2,1, figsize=(10, 8), sharey = True, sharex=True)
    
    x1 = list(sci_data.keys())[4:]
    y1 = [0]
    for i in range(1, len(x1)):
        y1.append((sum(sci_data[x1[i]]) - sum(sci_data[x1[i-1]]))/sum(sci_data[x1[i-1]]))
    ax[0].bar(x1, y1, color='blue')
    ax[0].legend(['Scientific'], loc='upper left')
    
    x2 = list(mis_data.keys())[4:]
    y2 = [0]
    for i in range(1, len(x2)):
        y2.append((sum(mis_data[x2[i]]) - sum(mis_data[x2[i-1]]))/sum(mis_data[x2[i-1]]))
    ax[1].bar(x2, y2, color='orangered')
    ax[1].legend(['Misinformation'], loc='upper left')
    
    ax[0].set_title('Growth Rate of Likes per year for both groups')
    plt.xlabel('Year')
    plt.ylabel('Rate of Growth')
    
    plt.savefig(outpath + '/normalized_growth_over_year.png', bbox_inches='tight')

def group_median_over_month(sci_inpath, misinfo_inpath, outpath):
    '''
    charts median number of likes for each month for each group  
    '''
    with open(sci_inpath) as f:
        sci_data = json.load(f)
        
    with open(misinfo_inpath) as f:
        mis_data = json.load(f)
        
    fig, ax = plt.subplots(2,1, figsize=(10, 8), sharey = True, sharex=True)
    
    x1 = list(sci_data.keys())[100:-7]
    y1 = []
    for year in x1:
        y1.append(np.median(sci_data[year]))
    ax[0].plot(x1, y1, color='blue')
    ax[0].legend(['Scientific'])
    
    x2 = list(mis_data.keys())[100:-7]
    y2 = []
    for year in x2:
        y2.append(np.median(mis_data[year]))
    ax[1].plot(x2, y2, color='orangered')
    ax[1].legend(['Misinformation'], loc='upper left')
    
    ax[0].set_title('Median number of likes for per month for both groups')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Median Number of Likes')
    
    plt.savefig(outpath + '/both_group_median_over_month.png', bbox_inches='tight')


def sci_likes_over_months(politicians, inpath, outpath):
    """
    takes in list of names for politicians and creates line
    graph for their likes over months
    * takes politician with highest tweet ratio average and 
    lowest tweet ratio average from scientific group
    """
    with open(inpath) as f:
        avg_likes_over_months = json.load(f)
    
    plt.figure(figsize=(12, 5))
    for i in politicians:
        plt.plot(list(avg_likes_over_months[i].keys()), list(avg_likes_over_months[i].values()))
        
        plt.title('Scientific Average likes Over 4 Months')
        plt.xlabel('Months')
        plt.xticks(rotation=90) 
        plt.ylabel('Number of Likes')
        plt.legend(politicians)

    plt.savefig(outpath + '/scientific_likes_over_months.png', bbox_inches='tight')

def misinfo_likes_over_months(politicians, inpath, outpath):
    """
    takes in list of names for politicians and creates line
    graph for their likes over months
    * takes politician with highest tweet ratio average and 
    lowest tweet ratio average from misinfo group
    """
    with open(inpath) as f:
        avg_likes_over_months = json.load(f)
    
    plt.figure(figsize=(12, 5))
    for i in politicians:
        plt.plot(list(avg_likes_over_months[i].keys()), list(avg_likes_over_months[i].values()))
        
        plt.title('Misinfo Average likes Over 4 Months')
        plt.xlabel('Months')
        plt.xticks(rotation=90) 
        plt.ylabel('Number of Likes')
        plt.legend(politicians)
        
    plt.savefig(outpath + '/misinfo_likes_over_months.png', bbox_inches='tight')

def compare_sci_misinfo(politicians, sci_path, misinfo_path, outpath):
    """
    Compares 1 politician from each group
    """
    with open(sci_path) as f:
        sci = json.load(f)

    with open(misinfo_path) as f:
        misinfo = json.load(f)

    plt.figure(figsize=(12, 5))
    plt.plot(list(sci[politicians[0]].keys()), list(sci[politicians[0]].values()))
    plt.plot(list(misinfo[politicians[1]].keys()), list(misinfo[politicians[1]].values()))

    plt.title('Comparison of Scientific and Misinfo Max Likes Over 4 Months Window')
    plt.xlabel('Months')
    plt.xticks(rotation=90) 
    plt.ylabel('Max Number of Likes')
    plt.legend(politicians)
    
    plt.savefig(outpath + '/compare_sci_misinfo.png', bbox_inches='tight')
    
    
def max_all_sci(sci_path, outpath):
    """
    returns max likes for all scientific politicians
    """
    with open(sci_path) as f:
        sci = json.load(f)

    plt.figure(figsize=(20, 10))
    for i in list(sci.keys()):
        plt.plot(list(sci[i].keys()), list(sci[i].values()))
        
        plt.title('Scientific Max Likes Over 4 Months')
        plt.xlabel('Months')
        plt.xticks(rotation=90) 
        plt.ylabel('Number of Likes')
        plt.legend(list(sci.keys()), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.savefig(outpath + '/max_all_sci.png', bbox_inches='tight')

    
def max_all_misinfo(misinfo_path, outpath):
    """
    returns max likes for all misinfo politicians
    """
    with open(misinfo_path) as f:
        misinfo = json.load(f)

    plt.figure(figsize=(20, 10))
    for i in list(misinfo.keys()):
        plt.plot(list(misinfo[i].keys()), list(misinfo[i].values()))
        
        plt.title('Misinformation Max Likes Over 4 Months')
        plt.xlabel('Months')
        plt.xticks(rotation=90) 
        plt.ylabel('Number of Likes')
        plt.legend(list(misinfo.keys()), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
    plt.savefig(outpath + '/max_all_misinfo.png', bbox_inches='tight')
