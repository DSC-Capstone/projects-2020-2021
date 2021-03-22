import pandas as pd
import numpy as np
import sklearn
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score


## helper function
def plot_TPR_FPR(FPR_lst, TPR_lst,outdir):
    
    fp1 = os.path.join(outdir, 'FPR_TPR.png')
    ## TPR vs. FPR
    plt.figure()
    plt.plot(FPR_lst,TPR_lst)
    plt.xlabel('False Positive Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)
    plt.title('TPR vs. FPR',fontsize=15)
    plt.savefig(fp1)
    plt.show()
    
    
    
## helper function
def plot_TPR_FDR(TPR_lst, FDR_lst,outdir):
    
    fp2 = os.path.join(outdir, 'TPR_FDR.png')
    ## TPR vs. FDR
    plt.figure()
    plt.plot(FDR_lst,TPR_lst)
    plt.xlabel('False Discovery Rate',fontsize=15)
    plt.ylabel('True Positive Rate',fontsize=15)
    plt.title('TPR vs. FDR',fontsize=15)
    plt.savefig(fp2)
    plt.show()

    
## helper function
def plot_rates(threshold, FPR_lst, TPR_lst, FDR_lst,outdir):  
    
    fp3 = os.path.join(outdir, 'rates.png')
    # line 1 points
    x1 = threshold
    y1 = FPR_lst
    
    plt.figure()
    
    # plotting the line 1 points 
    plt.plot(x1, y1, label = "False Positive Rate")


    # line 2 points
    x2 = threshold
    y2 = TPR_lst
    # plotting the line 2 points 
    plt.plot(x2, y2, label = "True Positive Rate")

    # line 3 points
    x3 = threshold
    y3 = FDR_lst
    # plotting the line 2 points 
    plt.plot(x3, y3, label = "False Discovery Rate")

    plt.xlabel('Threshold')
    # Set the y axis label of the current axis.
    plt.ylabel('Rate')
    # Set a title of the current axes.
    plt.title('Rates as function of Threshold')
    # show a legend on the plot
    plt.legend()
    # Display a figure.
    plt.savefig(fp3)
    plt.show()

    
def prediction_model(path,outdir):
    
    #df = pd.read_csv(path, sep=';')
    df = path
    X = df.drop(['cardio'], axis=1)
    y = df['cardio']

    
    FPR_lst = []
    TPR_lst = []
    FDR_lst = []
    scores = []
    threshold = np.arange(0.1,0.9,0.01)

    clf = LogisticRegression(class_weight="balanced")
    clf.fit(X, y)

    for i in threshold:

        preds = pd.Series(clf.predict_proba(X)[:,1])
        preds[preds >= i] = 1
        preds[preds < i] = 0

        TP_ = np.logical_and(preds, y)
        FP_ = np.logical_and(preds, np.logical_not(y))
        TN_ = np.logical_and(np.logical_not(preds), np.logical_not(y))
        FN_ = np.logical_and(np.logical_not(preds), y)

        TP = sum(TP_)
        FP = sum(FP_)
        TN = sum(TN_)
        FN = sum(FN_)

        FPR = FP/(FP+TN)
        FPR_lst.append(FPR)

        TPR = TP/(TP+FN)
        TPR_lst.append(TPR)

        FDR = FP/(FP+TP)
        FDR_lst.append(FDR)

        scores.append(accuracy_score(preds, y))
        
    plot_rates(threshold, FPR_lst, TPR_lst, FDR_lst,outdir)
    plot_TPR_FPR(FPR_lst, TPR_lst,outdir)
    plot_TPR_FDR(TPR_lst, FDR_lst,outdir)
    return pd.DataFrame(data={'threshold':threshold,'TPR':TPR_lst,'FPR':FPR_lst,'FDR':FDR_lst})


