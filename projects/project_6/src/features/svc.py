import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import os


def SVC_model(cleaned):
    X_train, X_vali, y_train, y_vali = train_test_split(cleaned['text'], cleaned['sentiment'], test_size = 0.25, random_state=0)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X_train)
    dictionary = vectorizer.get_feature_names()
    X_validation = vectorizer.transform(X_vali)
    X_train_bag_of_words_rep = X.toarray()
    X_vali_bag_of_words_rep = X_validation.toarray()
    clf = SVC(C = 0.1, kernel = 'linear', gamma = "auto")
    clf.fit(X,y_train)
    result = clf.predict(X_validation)
    accuracy = accuracy_score(result, y_vali)
    return accuracy, result, vectorizer, clf

def build_svc(**kwargs):
    path,cleanpath,outpath = kwargs['data_path'],kwargs['cleaned_csv'],kwargs['out_path']
    df = pd.read_csv(path+cleanpath)
    results = SVC_model(df)
    # print the accuracy
    print('accuracy of svc model on the labeled dataset:', results[0])

    # save the prediction
    if(kwargs['model'] == 'svc'):
        score_list,date_list = [],[]
        for i in os.listdir(path):
            if '-clean' in i:
                test_df = pd.read_csv(path+i)
                score = np.mean(results[3].predict(results[2].transform(test_df['clean_text'])))
                score_list.append(score)
                date_list.append(i[:10])
        df_score = pd.DataFrame({'date':date_list, "score":score_list})
        df_score['date'] = pd.to_datetime(df_score['date'])
        df_score.sort_values(by=['date'], inplace=True)
        if not os.path.exists(outpath):
            os.mkdir(outpath)
        df_score.to_csv(outpath+kwargs['score_csv'],index=False)
        print('predictions of SVC are saved in `data/final`')
    return