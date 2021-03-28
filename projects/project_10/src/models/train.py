import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import pickle

import logging

from src.utils import ensure_path_exists

def train_model(source, out, validation_size, classifier, model_params):
    """
    Trains model with data preparation and desired classifier.
    """

    df = pd.read_csv(source)
    X = df.drop(columns=['provider']).values
    y = df['provider'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=validation_size)

    #Normalizing data with respect to the entire training data
    scaler = StandardScaler()
    scaler.fit(X)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    #Choosing the desired model type
    classifier_type = {
        "RandomForest": RandomForestClassifier,
        "KNN": KNeighborsClassifier,
        "LogisticRegression": LogisticRegression
    }

    model_params = model_params[classifier]
    clf = classifier_type[classifier](**model_params)

    #Outputting prediction and accuracy score
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print('%s model accuracy: %s' % (classifier, accuracy))
    logging.info('%s model accuracy: %s' % (classifier, accuracy))
    
    # Plot non-normalized confusion matrix
    class_names = ['other', 'youtube', 'prime', 'netflix', 'yt-live', 'twitch']
    
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=None);
    disp.ax_.set_title("Confusion matrix, without normalization");
    plt.savefig('confusion_matrix.png');
    plt.show()
    
    disp = plot_confusion_matrix(clf, X_test, y_test,
                                 display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize='true');
    disp.ax_.set_title("Normalized confusion matrix");
    plt.savefig('normalized_confusion_matrix.png');
    plt.show()

    ensure_path_exists(out)
    with open(out, 'wb') as outfile:
        pickle.dump(clf, outfile)
    logging.info(f'Model saved to {out}')

    return clf