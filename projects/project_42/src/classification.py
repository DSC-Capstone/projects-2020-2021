import pandas as pd
import numpy as np
import nltk
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from nltk.stem import PorterStemmer 
import warnings
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn import svm
warnings.filterwarnings("ignore")



def build(summary,genre, text_feature = 1, baseline = 1,top_genre = 10,top_phrases = 10):
    """parameter tuned classify models"""
    #remove punctuation
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    no_punct = summary.apply(lambda x: tokenizer.tokenize(x))

    #label binarizer
    multilabel_binarizer = sklearn.preprocessing.MultiLabelBinarizer()
    multilabel_binarizer.fit(genre)
    label = multilabel_binarizer.transform(genre)
   
    #split training and validation set
    xtrain, xval, ytrain, yval = train_test_split(summary, label, test_size=0.2, random_state=1000)
    tfidf_vectorizer = TfidfVectorizer(analyzer = 'word', stop_words = {'english'}, max_df=0.8, max_features=10000)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
    xval_tfidf = tfidf_vectorizer.transform(xval)
    
  
    #hyperparameter grid search
    parameters = {
        "estimator__C":[0.1,1,5,10,15],
    }
    

    if baseline:
        lr = sklearn.linear_model.LogisticRegression()
        clf = OneVsRestClassifier(lr,n_jobs = 8)
        clf.fit(xtrain_tfidf,ytrain)
        y_pred = clf.predict(xval_tfidf)
    else:
        svc = svm.LinearSVC()
        clf = OneVsRestClassifier(svc,n_jobs = 8)
        clf = GridSearchCV(clf, param_grid=parameters,cv = 3, verbose = 3, scoring = 'f1_micro',refit = True)
        clf.fit(xtrain_tfidf,ytrain)
        y_pred = clf.predict(xval_tfidf)
        clf = clf.best_estimator_

    # Predicted label
    actual_genre = multilabel_binarizer.inverse_transform(yval)
    predicted_genre = multilabel_binarizer.inverse_transform(y_pred)
    
    
    #evaluation
    f1 = "f1-score: "+str(sklearn.metrics.f1_score(yval, y_pred, average="micro"))
    
    e1 = 'percentage of genres that are correctly predicted: '+ str(np.sum([len(set(a).intersection(b)) for a, b in \
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/sum(genre.apply(len)))
    e2 = 'percentage of movies that have at least one gnere predicted right: '+str(np.sum([len(set(a).intersection(b))>0 for a, b in\
                  zip(pd.Series(predicted_genre), pd.Series(actual_genre))])/len(genre))
    
    lst = []
    new_genre_label = []
    genre_label = multilabel_binarizer.classes_
    for a,b in zip(clf.estimators_, genre_label):
        try:
            lst.append(a.coef_)
            new_genre_label.append(b)
        except:
            pass

    dist = genre.explode().value_counts(ascending = False)
    genre_coef = dict(zip(new_genre_label,np.vstack(lst)))
    fig,ax = plt.subplots(top_genre//3+1,3,figsize = (20,top_genre*2))
    for o,g in enumerate(dist[:top_genre].index):
        c = genre_coef[g]
        words = tfidf_vectorizer.inverse_transform(c)[0]
        evd = [t for t in c if t >0]
        d = dict(zip(words,evd))
        sorted_words = sorted(d.items(), key=lambda item: item[1])[-top_phrases:]
        x = [i[0] for i in sorted_words]
        y = [i[1] for i in sorted_words]
        ax[o//3][o%3].barh(x,  y)
        ax[o//3][o%3].set_title(g)
    fig.tight_layout()
    if text_feature:
        if baseline:
            fig.savefig('data/figures/baseline model with summary text results.png')
        else:
            fig.savefig('data/figures/final model with summary text results.png')
    else:
        if baseline:
            fig.savefig('data/figures/baseline model with phrases results.png')
        else:
            fig.savefig('data/figures/final model with phrases results.png')
    return (f1+"\n"+e1+"\n"+e2+"\n")

def model(config):
    # read dataset path from config
    pikl = pd.read_pickle(config['data'])
    # read if run on baseline model from config
    baseline = config['baseline']
    # drop movie entries that do not have genres and add underscore to multigrams
    df = pikl.dropna(subset = ['genres'])
    df['genres']=df['genres'].apply(lambda x: np.nan if len(x)==0 else x)
    df = df.dropna(subset = ['genres'])
    df['phrases']=df['phrases'].apply(lambda a: [i.replace(' ','_') for i in a])
    
    ngenre = config['top_genre']
    nphrase = config['top_phrase']
    
    summary = df['summary']
    genre = df['genres']
    phrase = df['phrases'].apply(lambda x: ' '.join(x))
    

    s = build(summary,genre,1,baseline,ngenre,nphrase) # 1 means using text summary as feature
    p = build(phrase,genre,0,baseline,ngenre,nphrase)# 0 means using phrases as feature 

    print("=============Results=============")
    print('model performance using movie plot summary: '+ s +'\n' )
    print('model performance using phrases: '+p)
      
        
