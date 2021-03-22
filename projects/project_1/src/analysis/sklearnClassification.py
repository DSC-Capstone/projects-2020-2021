import pandas as pd
import numpy as np
import os
from sklearn import neighbors
# from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFdr, f_classif
from sklearn.metrics import confusion_matrix

datadir = 'data/raw/'
actual = pd.read_csv(datadir + 'actual.csv')
df_train = pd.read_csv(datadir + 'data_set_ALL_AML_train.csv')
df_test = pd.read_csv(datadir + 'data_set_ALL_AML_independent.csv')

#simple cleaning
#drop 'call'
to_drop = df_train.columns[df_train.columns.str.contains('call')]
df_train = df_train.drop(to_drop,axis=1)
to_drop = df_test.columns[df_test.columns.str.contains('call')]
df_test = df_test.drop(to_drop,axis=1)
#re-order the data: train
new_order = df_train.columns[2:].astype(int).sort_values().astype(str).tolist()
new_order.insert(0, 'Gene Description')
new_order.insert(1, 'Gene Accession Number')
df_train = df_train[new_order]
#re-order: test
new_order = df_test.columns[2:].astype(int).sort_values().astype(str).tolist()
new_order.insert(0, 'Gene Description')
new_order.insert(1, 'Gene Accession Number')
df_test = df_test[new_order]

#separate into x and y (gene expression and cancer type)
#let patients be rows and genes be features
x_train = df_train.iloc[:,2:].T
y_train = actual[actual.patient.isin(df_train.columns[2:].to_series().astype(int))].cancer.to_numpy()
x_test = df_test.iloc[:,2:].T
y_test = actual[actual.patient.isin(df_test.columns[2:].to_series().astype(int))].cancer.to_numpy()

#sklearn KNN classification: TODO: paramater tuning
knn = neighbors.KNeighborsClassifier()
knn.fit(x_train, y_train)
preds = knn.predict(x_test)
tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
fpr = fp / (fp + tn)
#TODO: comparison with empirical null analysis
#selectFdr: finds most import features based on specific attributes (f_classif: computes ANOVA f-value)
x_new = SelectFdr(f_classif, alpha=.01).fit_transform(x_train, y_train)
x_new = pd.DataFrame(x_new)
x_new.index+=1
#TODO: analysis w/ selectFdr genes