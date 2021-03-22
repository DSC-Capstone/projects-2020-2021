import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn import metrics
import joblib


def train(data_path, data_name, test_size, model, c, bagging, save_model, load_model):
    #retrieve  model
    if(load_model == 'True'):
        mdl = joblib.load(data_path + '/' + 'trained.model')
        return mdl

    #retrieve data
    file_path = data_path + "/"
    processed_data  = pd.read_csv(file_path + data_name)

    #pipelines
    processed_data = processed_data.drop(['guid', 'wait_msecs', 'chassistype', 'graphicscardclass'], axis = 1)
    
    #onehot norminal
    nominal = ['os', 'cpucode', 'persona']
    prepca = processed_data[nominal].fillna(method = 'backfill')
    pipe = OneHotEncoder()
    one_hot_system = pipe.fit(prepca).transform(prepca).todense()
    
    #combine quantative
    quan = ['before_cpuutil_max', 'before_harddpf_max', 'before_diskutil_max', 'ram', '#ofcores', 'age_category', 'processornumber']
    quan_features = processed_data[quan].values
    features = np.append(one_hot_system, quan_features, axis = 1)
    
    #PCA all
    pca=PCA(n_components=30)
    cols = pca.fit(features).transform(features)
    processed_df = processed_data[['ts','target']]
    index = 1
    for i in cols.transpose():
        processed_df['feature_' + str(index)] =  pd.Series(i, index = processed_data.index)
        index += 1

    #split train test
    train = processed_df[processed_df.ts.str[5:7] == '10'].drop('ts', axis = 1)
    test = processed_df[processed_df.ts.str[5:7] == '11'].drop('ts', axis = 1)

    if bagging == "True":
        #regroup
        train_small = train[(train.target == 2) | (train.target == 3)]
        train_1 = train[train.target == 1]
        train_1s = []
        sample_size = len(train) // 6
        for i in range(6):
            train_1s.append(train_1.sample(sample_size))
        
        #train model
        dts = []
        for i in range(6):
            temp = train_small.append(train_1s[i])
            model = DecisionTreeClassifier(max_depth = int(c)).fit(temp.drop('target', axis = 1), temp.target)
            dts.append(model)
        
        #test accuracy
        predictions = []
        for i in range(6):
            predictions.append(dts[i].predict(test.drop('target', axis = 1)))
        prediction = []
        for i in range(len(predictions[0])):
            judge = {1:0, 2:0, 3:0}
            for j in range(len(predictions)):
                judge[predictions[j][i]] += 1
            prediction.append(max(judge.items(),key=lambda x:x[1])[0])
        
        #report accuracy
        f = open(data_path + '/' + 'bagging_acc.txt', mode = 'w')
        print("---------------------------------------------------")
        print("Accuracy on Test Set:")
        print(metrics.classification_report(test.target, prediction, digits=3))
        f.write(metrics.classification_report(test.target, prediction, digits=3))
        f.close()
        
        #save model
        if(save_model == 'True'):
            for i in range(6):
                joblib.dump(dts[i], data_path + '/' + 'trained.model' + str(i))
        return dts

    else:    
        #choose the model
        mdl = None
        if(model == "decision tree"):
            mdl = DecisionTreeClassifier(max_depth = int(c))
        elif(model == "SVM"):
            mdl = SVC(C = int(c))
        else:
            return None

        #train model
        mdl = mdl.fit(train.drop('target', axis = 1), train.target)

        #report accuracy
        f = open(data_path + '/' + 'train_acc.txt', mode = 'w')
        prediction = mdl.predict(train.drop('target', axis = 1))
        print("---------------------------------------------------")
        print("Accuracy on Train Set:")
        print(metrics.classification_report(train.target, prediction, digits=3))
        f.write(metrics.classification_report(train.target, prediction, digits=3))
        f.close()
        f = open(data_path + '/' + 'test_acc.txt', mode = 'w')
        prediction = mdl.predict(test.drop('target', axis = 1))
        print("---------------------------------------------------")
        print("Accuracy on Test Set:")
        print(metrics.classification_report(test.target, prediction, digits=3))
        f.write(metrics.classification_report(test.target, prediction, digits=3))
        f.close()

        #save model
        if(save_model == 'True'):
            joblib.dump(mdl, data_path + '/' + 'trained.model')

        return mdl
    return