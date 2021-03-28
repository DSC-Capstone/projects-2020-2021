import pandas as pd
import matplotlib.pyplot as plt

import sys

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression


def build_classifier(x,y, classifier_list,output_path, trail_num=5 ):



    knn_param = list(range(2,50,5))
    dt_param = list(range(5,100,10))
    rf_param = list(range(5,100,10))
    nn_param = list(range(300,800,100))
    svm_param = [10**i for i in range(-3,4)]
    sgd_param = list(range(300,1100,100))
    l_param = [10**i for i in range(-3,4)]


    print('Start to build the classifier and tune the parameter')

    result = pd.DataFrame(columns =['trail','classifier','parameter','train_acc','test_acc','train_f1','test_f1'])

    for t in range(trail_num):
        #train test split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

        #KNN
        if 'knn' in classifier_list:
            print("Start KNN")
            #tune the parameter
            for i in knn_param:

                clf = KNeighborsClassifier(n_neighbors=i)
                clf.fit(x_train, y_train)

                #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='macro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='macro')

                temp = {'trail':t,
                        'classifier':'KNN',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)


        #DecisionTree
        if 'decision tree' in classifier_list:
            print("Start Decision Tree")
            for i in dt_param:
                clf = DecisionTreeClassifier(max_depth= i)
                clf.fit(x_train, y_train)

                 #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='micro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='micro')

                temp = {'trail':t,
                        'classifier':'Decision Tree',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)

        #Random Forest
        if 'random forest' in classifier_list:
            print("Start Random Forest")
            for i in rf_param:
                clf = RandomForestClassifier(max_depth =i)
                clf.fit(x_train, y_train)

                #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='macro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='macro')

                temp = {'trail':t,
                        'classifier':'Random Forest',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)


        #neural network
        if 'neural network' in classifier_list:
            print("Start Neural network")
            for i in nn_param:
                clf = MLPClassifier( max_iter=i)
                clf.fit(x_train, y_train)

                #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='micro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='micro')

                temp = {'trail':t,
                        'classifier':'Neural network',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)


        #SGD
        if 'sgd' in classifier_list:
            print("Start SGD")
            for i in sgd_param:
                clf = SGDClassifier( max_iter = i)
                clf.fit(x_train, y_train)

                #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='micro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='micro')

                temp = {'trail':t,
                        'classifier':'SGD',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)


        #logistic
        if 'logistic' in classifier_list:
            print("Start logistic")
            for i in l_param:
                clf = LogisticRegression( C = i)
                clf.fit(x_train, y_train)

                #get train score
                y_train_pred = clf.predict(x_train)
                train_sc = accuracy_score(y_train, y_train_pred)
                train_f1 = f1_score(y_train, y_train_pred,average='micro')

                #get test score
                y_test_pred = clf.predict(x_test)
                test_sc = accuracy_score(y_test, y_test_pred)
                test_f1 = f1_score(y_test, y_test_pred,average='micro')

                temp = {'trail':t,
                        'classifier':'logistic',
                        'parameter': i,
                        'train_acc':train_sc,
                        'test_acc':test_sc,
                        'train_f1':train_f1,
                        'test_f1':test_f1}
                print(temp)
                result = result.append(temp,ignore_index=True)

    print("All Done!")
    result.to_csv(output_path, index = False)
    return result
