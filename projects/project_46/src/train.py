import pickle
import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

def train(train_config):

    print()
    print('===================================================================')
    print(' => Training models...')
    print()

    data = pd.read_pickle(train_config['data_dir'] + 'processed/feature_encoded_merged_data.pkl')

    # n_phrases = len(data['unigram_vec'].values[0])
    #
    # def select_phrases(phrases, n_phrases):
    #     return phrases[:n_phrases]
    #
    # data['top_phrases'] = data['phrase_vec'].apply(lambda x: select_phrases(x, n_phrases))

    train = data.loc[data['dataset'] == 'train'].copy()
    test = data.loc[data['dataset'] == 'test'].copy()

    def create_model(all_data, train, **kwargs):

        num_train = train[['Surprise(%)', 'price_change_7',
                  'price_change_30', 'price_change_90', 'price_change_365',
                  'prev_vix_values']].to_numpy()

        scaler = StandardScaler()
        scaler.fit(num_train)
        num_train = scaler.transform(num_train)

        mlb = MultiLabelBinarizer()
        all_events = pd.DataFrame(mlb.fit_transform(all_data['cleaned_event']),
                                  columns = mlb.classes_,
                                  index = all_data['cleaned_event'].index)
        train_events = all_events.iloc[all_data.loc[all_data['dataset'] == 'train'].index].to_numpy()

        train_y = train[['target']].to_numpy().ravel()

        if kwargs['train_type'] == 'unigram':
            train_unigrams = np.array(train['unigram_vec'].values.tolist())
            train_X = np.concatenate((train_events, num_train, train_unigrams), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000, max_features = kwargs['max_features'])
            model = model.fit(train_X, train_y)

        if kwargs['train_type'] == 'phrase':
            train_phrases = np.array(train['top_phrases'].values.tolist())
            train_X = np.concatenate((train_events, num_train, train_phrases), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000, max_features = kwargs['max_features'])
            model = model.fit(train_X, train_y)

        if kwargs['train_type'] == 'base':
            train_X = np.concatenate((train_events, num_train), axis = 1)

            model = RandomForestClassifier(max_depth = 10, n_estimators = 2000)
            model = model.fit(train_X, train_y)

        return model


    if train_config['testing']:
        max_features = 50
    else:
        max_features = 1250
    print('  => Training baseline model...')
    print()

    base_model = create_model(data, train, train_type = 'base', max_features = max_features)

    print()
    print('  => Training unigram model...')
    print()

    uni_model = create_model(data, train, train_type = 'unigram', max_features = max_features)

    print()
    print('  => Training phrase model...')
    print()

    phrase_model = create_model(data, train, train_type = 'phrase', max_features = max_features)

    print()
    print('  => Exporting test results to pkl...')
    print()

    num_train = train[['Surprise(%)', 'price_change_7',
               'price_change_30', 'price_change_90', 'price_change_365',
               'prev_vix_values']].to_numpy()

    test_scaler = StandardScaler()

    num_test = test[['Surprise(%)', 'price_change_7',
                   'price_change_30', 'price_change_90', 'price_change_365',
                   'prev_vix_values']].to_numpy()

    test_scaler.fit(num_train)
    num_test = test_scaler.transform(num_test)

    mlb = MultiLabelBinarizer()
    all_events = pd.DataFrame(mlb.fit_transform(data['cleaned_event']),
                                  columns = mlb.classes_,
                                  index = data['cleaned_event'].index)

    test_events = all_events.iloc[data.loc[data['dataset'] == 'test'].index].to_numpy()
    test_unigrams = np.array(test['unigram_vec'].values.tolist())
    test_phrase = np.array(test['top_phrases'].values.tolist())

    base_test_X = np.concatenate((test_events, num_test), axis = 1)
    unigram_test_X = np.concatenate((test_events, num_test, test_unigrams), axis = 1)
    phrase_test_X = np.concatenate((test_events, num_test, test_phrase), axis = 1)

    test_y = test[['target']].to_numpy().ravel()

    test['base_pred'] = base_model.predict(base_test_X)
    test['unigram_pred'] = uni_model.predict(unigram_test_X)
    test['phrase_pred'] = phrase_model.predict(phrase_test_X)

    # Exporting to local pickle file
    os.system('mkdir -p ' + train_config["data_dir"] + 'tmp/')
    test.to_pickle(train_config["data_dir"] + 'tmp/model_results.pkl')

    # Saving models
    print()
    print('  => Exporting models to pkl...')
    print()

    out_dir = train_config["data_dir"] + train_config['output_file']
    os.system('mkdir -p ' + out_dir)

    with open(out_dir + 'base_model', 'wb') as f:
        pickle.dump(base_model, f)
    with open(out_dir + 'uni_model', 'wb') as f:
        pickle.dump(uni_model, f)
    with open(out_dir + 'phrase_model', 'wb') as f:
        pickle.dump(phrase_model, f)
