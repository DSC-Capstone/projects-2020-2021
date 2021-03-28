
from typing import DefaultDict
import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns
import numpy as np
from astropy.visualization import hist
from wordcloud import WordCloud, STOPWORDS 
from string import punctuation
from collections import Counter

def draw_hist(outdir,pd_series,name,bins_type):
    '''
    Draw histogram without outliers
    '''
    Q1=pd_series.quantile(.25)
    Q3=pd_series.quantile(.75)
    IQR=1.5*(Q3-Q1)
    ax = plt.gca()
    hist(pd_series[pd_series.between(Q1-IQR, Q3+IQR)],bins=bins_type,ax=ax, density=True)
    ax.grid(color='grey', alpha=0.5, linestyle='solid')
    plt.savefig(os.path.join(outdir, name))


def draw_barh(outdir,df):
    '''
    Draw barh graph on the ner tag.
    '''
    ner_tag = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    tag_count = DefaultDict(int)
    for i in df.ner_tags:
        for j in i:
            tag_count[ner_tag[j]] += 1        
    df_tag_count = pd.DataFrame.from_dict(tag_count, 'index').sort_values(by =0)
    df_tag_count.columns = ['counts']
    df_tag_count.iloc[:-1,:].plot.barh()
    plt.savefig(os.path.join(outdir, 'barh.png'))


def phrase_cloud(outdir,df):
    '''
    create phrase cloud using the phrases extracted from dataset.
    '''
    token_collection={0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[]}
    for l,i in enumerate(df.ner_tags):
        phrase=''
        j=0
        k=0
        while j < len(i):
            phrase+=df.tokens.iloc[l][j]
            if i[j]!=0 and j<=len(i)-2 and i[j]==i[j+1]:
                    k=j
                    while k<=len(i)-2:
                        if i[k]==i[k+1]:
                            phrase=phrase+'_'+df.tokens.iloc[l][k+1]
                            k+=1
                        else:
                            break
                    token_collection[i[j]].append(phrase.lower())
                    phrase=''
                    j=k+1
            else:
                token_collection[i[j]].append(phrase.lower())
                j+=1
                phrase=''
    st=set(punctuation)|set(STOPWORDS)
    for i in range(9):
        token_collection[i]=[j for j in token_collection[i] if j not in st]

    stopwords = set(STOPWORDS) 

    for i in range(9):
        text =Counter(token_collection[i])
        wordcloud  = WordCloud(width = 400, height = 400, 
                        background_color ='white', 
                        stopwords = stopwords,
                        collocations = False,
                        min_font_size = 10).generate_from_frequencies(text) 
        plt.figure(figsize = (4,4), facecolor = None) 
        plt.title(ner_tag[i])

        plt.imshow(wordcloud) 
        plt.axis("off") 
        plt.tight_layout(pad = 0) 
        plt.savefig(os.path.join(outdir, ner_tag[i]+'.png'))
        plt.show() 
        
def generate_stats(outdir,**kwargs):
    '''
    Generate all the plots for eda
    '''
    os.makedirs(outdir, exist_ok=True)
    parent_dir=os.getcwd()
    dataset = load_dataset("conll2003")
    df=pd.concat([pd.DataFrame(dataset['train']),(pd.DataFrame(dataset['validation'])),(pd.DataFrame(dataset['test']))])
    df['sentence_length'] = df.tokens.apply(lambda x: len(x))

    
    
    hist_1=parent_dir+kwargs['hist_sentence_len']
    hist_2=parent_dir+kwargs['hist_ratio']
    barh=parent_dir+kwargs['barh']
    phrase=parent_dir+kwargs['phrase']
    
    draw_hist(hist_1,df.sentence_length,'hist_sentence_len','blocks')
    draw_hist(hist_2,df.ner_tags.apply(lambda x: len(np.nonzero(x)[0])/len(x)),'hist_ratio','freedman')
    draw_barh(barh,df)
    phrase_cloud(phrase,df)

