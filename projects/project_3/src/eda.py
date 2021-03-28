import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import ast

from collections import Counter
from wordcloud import WordCloud
from utils import get_project_root


root = get_project_root()
raw_data_path = os.path.join(root, 'data', 'raw', 'news')
processed_data_path = os.path.join(root, 'data', 'processed', 'news')
graph_data_path = os.path.join(root, 'data', 'graphs', 'news')

test_data_path = os.path.join(root, 'test', 'testdata')

def json_to_df(fp):
    """Reads in jsonl file and returns Pandas dataframe"""
    data = []
    with open(fp) as f:
        for line in f:
            data.append(json.loads(line))
            
    return pd.DataFrame(data)

def get_hashtags(df):
    """Returns a dictionary of hashtag occurences from dataframe of tweets"""
    
    def flatten(x):
        if len(x) != 0:
            return [d['text'] for d in x]
        else:
            return None
        
    inner_data = pd.json_normalize(df['entities'])
    hashtags = inner_data['hashtags'].apply(flatten)
    hashtags = hashtags.dropna()
    my_dict = Counter()

    for tag_list in hashtags:
        for tag in tag_list:
            my_dict[str.lower(tag)] +=1
    
    return my_dict

def count_hashtags(filename):
    """Returns dictionary of hashtag occurences from all files in folder"""
    total_counter = Counter()
    print()

    print('Parsing current file: ', filename)
    df = json_to_df(filename)

    hashtags_dict = get_hashtags(df)
    # try:
    #     hashtags_dict = get_hashtags(df)
    # except Exception as e:
    #     print(e)

    total_counter += hashtags_dict

    with open(processed_data_path + '/' + os.path.basename(filename) + '_top_100_hashtags.txt', "w", encoding='utf-8') as f:
        for k,v in  total_counter.most_common(100):
            # print(k,v)
            f.write( "{} {}\n".format(k,v) )
    return total_counter

def generate_word_cloud(counts, label, data_path):
    """Generates word cloud graph from dictionary of hashtag frequencies"""
    wordcloud = WordCloud(max_words=50, background_color="white")
    wordcloud.generate_from_frequencies(frequencies=counts)

    label = label[:-7]

    
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(label + 'Wordcloud')
    plt.savefig(data_path + '/' + label + '_wordcloud.png',bbox_inches='tight')

def plot_hashtag_counts_r(counter, label, data_path):
    most_common = counter.most_common(25)
    
    plt.figure(figsize=(20,15))
    plt.yticks(fontsize=25)
    plt.xticks(fontsize=20)
    plt.title('Top 25 Used Hashtags in ' + label + ' users', fontsize=30)
    plt.barh(y = [x[0] for x in most_common[::-1]], width = [x[1] for x in most_common[::-1]], color='r')
    plt.savefig(data_path + '/' + label + '_hashtag_counts.png')

def main(test=False):
    if test:
        data_path = test_data_path
    else:
        data_path = raw_data_path

    for file_ in os.listdir(data_path):
        if '_users.jsonl' in file_:
            # print(file_)
            counts = count_hashtags(data_path + '/' + file_)
            label = os.path.basename(file_)
            generate_word_cloud(counts, label, os.path.join(root, 'test', 'testreport'))
            plot_hashtag_counts_r(counts, label, os.path.join(root, 'test', 'testreport'))