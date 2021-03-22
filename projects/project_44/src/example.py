import subprocess
import os
import pandas as pd
import numpy as np
from IPython.display import Image
import re
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import json


def example(save_path,direct_path,raw_path,sample,segmentation, words):
    df = pd.read_csv(save_path + 'AutoPhrase_multi-words.txt',sep='\t',header=None)
    df.columns = ['Score','Words']
    subset = df[df['Score']>0.5].sample(100,replace=True)
    subset.sort_values('Score',ascending=False,inplace=True)
    subset.to_csv(save_path + "sample.txt", index=None, header = None, sep='\t')

    subset = pd.read_csv(direct_path + sample ,sep='\t',header=None)
    subset.columns = ['Score','Words','label']

    recalls =[]
    precisions =[]
    for i in np.arange(0.6,0.9,0.01):
        subset['Predicted'] = subset['Score'].apply(lambda x:1 if x>i else 0)
        TP = len(subset[(subset['label']==1)&subset['Predicted']==1])
        TN = len(subset[(subset['label']==0)&subset['Predicted']==0])
        FP = len(subset[(subset['label']==0)&subset['Predicted']==1])
        FN = len(subset[(subset['label']==1)&subset['Predicted']==0])
        recalls.append(TP/(TP+FP))
        precisions.append(TP/(TP+FN))
    plt.figure()
    plt.plot(recalls,precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision-recall curve')
    plt.savefig(direct_path+'precision-recall.png')
    plt.close()

    f = open(save_path + segmentation, "r")
    text = f.read()
    tokens = list(map(lambda y: list(map(lambda x:x[8:-9].replace(' ','_'),re.findall('<phrase>[^<]+</phrase>',y)))+list(filter(lambda x:len(x)>0,re.sub('<phrase>[^<]+</phrase>','',y).split(' '))),text.split('\n')))
    model = Word2Vec(tokens,workers=9)
    model.save(direct_path+'phrase_embedding.model')
    w = model.wv.vocab.keys()
    fp = open(direct_path + "word.txt", "w", encoding="utf-8")
    for word in w:
        fp.write(word + '\n')
    fp.close()

    sample3 = words.split(',')

    try:
        r = model.wv.most_similar(positive=sample3,topn=5)
        fp = open(direct_path + "most_similar.txt", "w", encoding="utf-8")
        for word in r:
            fp.write(word[0] + ' ' + str(word[1]) +'\n')
        fp.close()
    except:
        print('Try to change the word!')
    print("All Done! Check the similar words in the data/outputs/example!")
    return
