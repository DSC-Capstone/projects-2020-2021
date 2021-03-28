#!/usr/bin/env python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def ttsplit(X, y):
    return train_test_split(X, y, test_size=0.33, random_state=11)

def baseline_predict(X):
    return X.apply(lambda x:1 if x.num_smali >= 50000 else 0, axis=1)

def build_Log(X_train, y_train, C, max_it):
    return LogisticRegression(fit_intercept=True, C=C, max_iter=max_it).fit(X_train, y_train)

def build_DT(X_train, y_train, n):
    return DecisionTreeClassifier(max_depth = n).fit(X_train, y_train)

def build_KN(X_train, y_train, n):
    return KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)

def advanced_predict(reg, x):
    if x.self < x.java * 0.03:
        return 1
    if sum(x) < 1500 or x.kotlin > 0 or x.androidx > 0:
        return 0
    res = reg.predict([x.values])
    if res:
        if x.self > (x.java - x.java * 0.05) or x.self < (x.java + x.java * 0.05):
            return 1
        return 0
    else:
        return 0

def accuracy(y_train, y_test, y_predict_train, y_predict_test):
    print('train Accuracy = {}, test Accuracy = {}'.format(accuracy_score(y_train, y_predict_train),
                                                 accuracy_score(y_test, y_predict_test)))

def val_plot(X_train, X_test, y_train, y_test, n):
    train_accu = []
    test_accu = []
    for i in range(1,n):
        reg = build_KN(X_train, y_train, i)
        pred_tr = reg.predict(X_train)
        pred_te = reg.predict(X_test)
        train_accu += [accuracy_score(y_train, pred_tr)]
        test_accu += [accuracy_score(y_test, pred_te)]
    res = pd.DataFrame([train_accu,test_accu]).T
    res.columns = ['Train', 'Test']
    res.index = range(1,n)
    res.plot(kind='line')
    return res.Test.sort_values(ascending=False)[:5]
