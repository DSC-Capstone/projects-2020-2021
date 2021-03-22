import pandas as pd
import gzip
import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, 'src/data')
from Loading_Data import * 
sys.path.insert(0, 'src/eda')
from feature_selection import * 

# Helper Method1
# calculates the mean absolute error of the given model
def mae(model, X_train, y_train, X_test, y_test):
    reg = model.fit(X_train, y_train)
    train_error = mean_absolute_error(y_train, reg.predict(X_train))
    test_error = mean_absolute_error(y_test, reg.predict(X_test))
    
    return train_error, test_error

# run a simulation to see how different the observed mean absolute error is from the simulation 
def simulation2(model1, model2, X, y):
    errors1 = []
    errors2 = []
    
    for _ in range(1000):
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.3)

        reg1 = model1.fit(X_train2, y_train2)
        reg2 = model2.fit(X_train2, y_train2)
        
        errors1.append(mean_absolute_error(y_test2, reg1.predict(X_test2)))
        errors2.append(mean_absolute_error(y_test2, reg2.predict(X_test2)))
        
        
    return (np.array(errors1) - np.array(errors2))

# # the features are number of processes, page faults, capacity, cpu percentage, and cpu temperature 
# def train_test_XY(num_proc, page_faults, capacity, cpu_percent, cpu_temp, num_devices,avg_memory,cpu_sec):

#     X = pd.concat([num_proc, page_faults, capacity, cpu_percent, cpu_temp, num_devices,avg_memory,cpu_sec], axis = 1).dropna()
#     y = battery_event[['guid', 'battery_minutes_remaining']][battery_event.guid.isin(X.index)].groupby('guid')['battery_minutes_remaining'].apply(lambda x: (x!=-1).mean())

#     X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.3)
#     return X,y, X_train1, X_test1, y_train1, y_test1



##### Baseline Model #####
def linear_reg(X_train1, y_train1, X_test1, y_test1):
    # for our baseline model, we will use linear regression for calculating mean absolute error 
    linear_train, linear_test = mae(LinearRegression(), X_train1, y_train1, X_test1, y_test1)
    print("linear training error: " + str(linear_train), "linear test error: " + str(linear_test))
    
    return linear_train, linear_test
   


##### Improving Model #####
def supportvm(X_train1, y_train1, X_test1, y_test1):
    # to improve our baseline model, we will consider SVM for calculating mean absolute error 
    svm_train, svm_test = mae(svm.SVR(), X_train1, y_train1, X_test1, y_test1)
    print("svm training error: " + str(svm_train), "svm test error: " + str(svm_test))
    
    return svm_train, svm_test

    
def dtr(X_train1, y_train1, X_test1, y_test1):
    # This time, we will use decision tree regressor to calculate mean absolute error 
    dt_train, dt_test = mae(DecisionTreeRegressor(), X_train1, y_train1, X_test1, y_test1)
    print("decision tree training error: " + str(dt_train), "decision tree test error: " + str(dt_test))
    
    return dt_train, dt_test

def rf(X_train1, y_train1, X_test1, y_test1):
    rf_train, rf_test = mae(RandomForestRegressor(), X_train1, y_train1, X_test1, y_test1)
    print("random forest error: " + str(rf_train), "random forest test error: " + str(rf_test))
    
    return rf_train,rf_test

def ada(X_train1, y_train1, X_test1, y_test1):
    ada_train, ada_test = mae(AdaBoostRegressor(), X_train1, y_train1, X_test1, y_test1)
    print("adaBoosting error: " + str(ada_train), "adaBoosting test error: " + str(ada_test))
    
    return ada_train, ada_test

def gradient(X_train1, y_train1, X_test1, y_test1):
    gradient_train, gradient_test = mae(GradientBoostingRegressor(), X_train1, y_train1, X_test1, y_test1)
    print("gradient boosting error: " + str(gradient_train), "gradient boosting test error: " + str(gradient_test))

    return gradient_train, gradient_test
    
##### Hypothesis Testing #####

# Hypothesis Testing1
# Null Hypo: There's no difference in performance between Gradient Boosting Regressor and SVM
# Alternative Hypo: Gradient Boosting Regressor performs better than SVM
def hypo1(X,y,gradient_test,svm_test):
    print("Null Hypo: There's no difference in performance between Gradient Boosting Regressor and SVM")
    print("Alternative Hypo: Gradient Boosting Regressor performs better than SVM")
    observed_gradient_svm = gradient_test - svm_test
    print("Observed difference between gradient boosting error and svm error: " + str(observed_gradient_svm))
    
    diffa = simulation2(GradientBoostingRegressor(), svm.SVR(), X, y)
    p_gradient_svm = (diffa<observed_gradient_svm).mean()
    print("p-value: " + str(p_gradient_svm))
    
    
# Hypothesis Testing2
# Null Hypo: There's no difference in performance between Gradient Boosting Regressor and AdaBoosting Regressor
# Alternative Hypo: Gradient Boosting Regressor performs better than AdaBoosting Regressor
def hypo2(X,y,gradient_test,ada_test):   
    print("Null Hypo: There's no difference in performance between Gradient Boosting Regressor and AdaBoosting Regressor")
    print("Alternative Hypo: Gradient Boosting Regressor performs better than AdaBoosting Regressor")
    observed_gradient_ada = gradient_test - ada_test
    print("Observed difference between gradient boosting error and adaBoosting error: " + str(observed_gradient_ada))
    
    diffb = simulation2(GradientBoostingRegressor(), AdaBoostRegressor(), X, y)
    p_gradient_ada = (diffb<observed_gradient_ada).mean()
    print("p-value: " + str(p_gradient_ada))
    
    

