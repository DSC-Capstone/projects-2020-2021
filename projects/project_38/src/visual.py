import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from collections import Counter
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
from textblob import TextBlob
import re

def visual(out_dir,input_path,file,autophrase,multi_word,single_word,token_mapping):
    print("Creating Distribution graphs of the outputs")
    data_kk_single = pd.read_csv(input_path+single_word, sep="\t",header = None,names=['value', 'phrase'])
    data_kk_multi = pd.read_csv(input_path+multi_word, sep="\t",header = None,names=['value', 'phrase'])

    #single-word distribution
    plt.figure()
    plt.title('Quality Score Distribution of Single-Word Phrases')
    plt.hist(data_kk_single['value'], alpha=0.5, label='single', color='gray')
    plt.xlabel('Probability of Quality Word/Phrase')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')
    plt.savefig(input_path + 'single_value_distribution'+'.png')
    plt.close()

    #multi-word distribution
    plt.figure()
    plt.title('Quality Score Distribution of Multi-Word Phrases')
    plt.hist(data_kk_multi['value'], alpha=0.5, label='multi', color='red')
    plt.xlabel('Probability of Quality Word/Phrase')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')
    plt.savefig(input_path + 'multi_value_distribution' + '.png')
    plt.close()

    plt.figure()
    plt.title('Quality Score Distribution of Single-Word vs. Multi-Word Phrases')
    plt.hist(data_kk_single['value'], alpha = 0.5, label = 'single', color = 'gray')
    plt.hist(data_kk_multi['value'], alpha = 0.5, label = 'multi', color = 'red')
    plt.xlabel('Probability of Quality Word/Phrase')
    plt.ylabel('Frequency')
    plt.legend(loc = 'upper right')
    plt.savefig(input_path + 'comparison_quality_score' + '.png')
    plt.close()

    try:
        with open('data/raw/DBLP.5K.txt', 'r') as file:
            data = file.read().split('\n')

        #sentiment analysis
        kk = []
        for i in data:
            kk.append(TextBlob(i).sentiment.polarity)
        plt.figure()
        plt.hist(kk, color='green')
        plt.title('Sentiment Polarity Distribution')
        plt.xlabel('Polarity Score')
        plt.ylabel('Frequency')
        plt.savefig(input_path+ 'sentiment_polarity_distribution'+'.png')
        plt.close()

        dd = []
        for i in data:
            dd.append(TextBlob(i).sentiment.subjectivity)
        plt.figure()
        plt.hist(dd, color='yellow')
        plt.title('Sentiment Subjectivity Distribution')
        plt.xlabel('Subjectivity Score')
        plt.ylabel('Frequency')
        plt.savefig(input_path+ 'sentiment_subjectivity_distribution'+'.png')
        plt.close()


        #tf-idf top 20 single
        tfIdfVectorizer=TfidfVectorizer(stop_words='english')
        tfIdf = tfIdfVectorizer.fit_transform(data)
        cum = []
        for i in tqdm(range(tfIdf.shape[0])):
            df = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
            df = df[df['TF-IDF']!=0].sort_values('TF-IDF', ascending=False)
            cum.append(df['TF-IDF'].to_dict())
        counter = collections.Counter()
        for d in cum:
            counter.update(d)
        res = pd.DataFrame({'Word':dict(counter).keys(),'Score':dict(counter).values()})
        res['Score'] = res['Score'].apply(lambda x: (x-min(res['Score']))/(max(res['Score'])-min(res['Score'])))
        res.sort_values('Score',ascending=False,inplace=True)
        res.to_csv('data/outputs/tfidfsingle.csv',index=False)

        #autophrase top 20 single
        ds = pd.read_csv('data/outputs/AutoPhrase_single-word.txt',sep='\t')
        row = ds.columns.values
        ds.columns = ['Score','Word']
        ds.loc[len(df)] = row
        ds['Score'] = ds['Score'].apply(lambda x:float(x))
        ds.sort_values('Score',inplace=True,ascending=False)
        res.index = res.Word
        ds.index = ds.Word
        ds.to_csv('data/outputs/qualitysingle.csv',index=False)

        #multiplication top 20 single
        haha = {}
        for key in ds.Score.to_dict():
            try:
                value = (ds.Score.to_dict()[key] * res.Score.to_dict()[key])
                haha[key] = value
            except:
                pass
        wala = pd.DataFrame({'Word':haha.keys(),'Score':haha.values()})
        wala.sort_values('Score',ascending=False,inplace=True)
        wala.to_csv('data/outputs/multiplicationsingle.csv',index=False)


        #tf-idf top 20 multi
        tfIdfVectorizer=TfidfVectorizer(stop_words='english',ngram_range=(2,3))
        tfIdf = tfIdfVectorizer.fit_transform(data)
        cum = []
        for i in tqdm(range(tfIdf.shape[0])):
            df = pd.DataFrame(tfIdf[i].T.todense(), index=tfIdfVectorizer.get_feature_names(), columns=["TF-IDF"])
            df = df[df['TF-IDF']!=0].sort_values('TF-IDF', ascending=False)
            cum.append(df['TF-IDF'].to_dict())
        counter = collections.Counter()
        for d in cum:
            counter.update(d)
        res = pd.DataFrame({'Word':dict(counter).keys(),'Score':dict(counter).values()})
        res['Score'] = res['Score'].apply(lambda x: (x-min(res['Score']))/(max(res['Score'])-min(res['Score'])))
        res.sort_values('Score',ascending=False,inplace=True)
        res.to_csv('data/outputs/tfidfmulti.csv',index=False)


        #autophrase top 20 multi
        ds = pd.read_csv('data/outputs/AutoPhrase_multi-words.txt',sep='\t')
        row = ds.columns.values
        ds.columns = ['Score','Word']
        ds.loc[len(df)] = row
        ds['Score'] = ds['Score'].apply(lambda x:float(x))
        ds.sort_values('Score',inplace=True,ascending=False)
        res.index = res.Word
        ds.index = ds.Word
        ds.to_csv('data/outputs/qualitymulti.csv',index=False)


        #multiplication top 20 multi
        haha = {}
        for key in ds.Score.to_dict():
            try:
                value = (ds.Score.to_dict()[key] * res.Score.to_dict()[key])
                haha[key] = value
            except:
                pass
        lala = pd.DataFrame({'Word':haha.keys(),'Score':haha.values()})
        lala.sort_values('Score',ascending=False,inplace=True)
        lala.to_csv('data/outputs/multiplicationmulti.csv',index=False)


        #combine all
        final = pd.concat([wala,lala]).sort_values('Score',ascending=False)
        final['Score'] = final['Score'].apply(lambda x:round(x,2))
        final.to_csv('data/outputs/multiplicationall.csv',index=False)


        # scatterplot - frequency vs score
        with open('data/raw/DBLP.5K.txt', 'r') as file_op:
            d = file_op.read()

        word_list = re.findall(r'[A-Za-z]+[0-9]?[+-]*', d)
        word_count = Counter(word_list)
        single_top_20 = data_kk_single[:20]
        single_top_20['frequency'] = single_top_20.apply(lambda row: word_count[row['phrase']], axis = 1)
        single_top_20.to_csv('data/outputs/singletop20.csv', index=False)

        plt.title('Word Frequency vs AutoPhrase Score')
        plt.xlabel('Score')
        plt.ylabel('Frequency')

        for index, row in single_top_20.iterrows():
            plt.text(x = row.value,
                    y = row.frequency,
                    s = row.phrase,
                    size = 5,
                    horizontalalignment = 'center')

        plt.scatter(x = single_top_20.value,
                    y = single_top_20.frequency,
                    c = single_top_20.frequency,
                    s = single_top_20.value,
                    linewidths = 2,
                    edgecolor='w',
                    alpha = 0.5)

        plt.savefig(input_path + 'freq_score_plot' + '.png', dpi = 300)
        plt.close()

    except:
        print('This part works for test run only!')


    print("Done!")
    print("Distribution graphs are in the /data/outputs folder!")
    return
