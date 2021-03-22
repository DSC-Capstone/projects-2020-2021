import json
import os
import sys
import numpy as np
from src.preprocessing import bbc_preprocessing, news_preprocessing
import src.model as model
sys.path.insert(0, 'src')
from src import ner_eda as eda
import src.utils as utils
import pickle




def main(targets):
    # Load the AutoPhrase module
    os.system('git submodule init && git submodule update')

    if 'all' in targets:
        targets = 'all preprocessing autophrase model'
        para = json.load(open('config/all_cfg.json'))
        
    if 'test' in targets:
        targets = 'test preprocessing autophrase model'
        para = json.load(open('config/all_cfg.json'))

    if 'preprocessing' in targets:
        print("Preprocessing 20 News Dataset")
        os.makedirs('data/temp',exist_ok=True)
        news_preprocessing('data/temp')
        
    if 'autophrase' in targets:
        print("Running AutoPhrase")
        auto_para = ' '.join([f'{param}={value}' for param, value in json.load(open('config/autophrase_cfg.json')).items()])
        os.system(f'cd AutoPhrase/ && {auto_para} ./auto_phrase.sh')
        os.system(f'cd ..')
        
        
        
    if 'model' in targets:
        if 'all' in targets:
            print("Load Data")
            newsgroups_train_X,newsgroups_test_X,newsgroups_val_X, newsgroups_train_y,newsgroups_test_y,newsgroups_val_y = utils.load_20_news()
        if 'test' in targets:
            print("Load Data")
            newsgroups_train_X,newsgroups_test_X,newsgroups_val_X, newsgroups_train_y,newsgroups_test_y,newsgroups_val_y = utils.load_20_news_test(500)
        
        print("Build the model")
        if para["feature"] == "all":
            autophrase = utils.phrase_preprocess(para['autophrase_result'])
            ner = utils.ner_preprocess(para["ner_result"])
            vocab_lst = autophrase + ner
        elif para["feature"] == "ner":
            vocab_lst = utils.ner_preprocess(para["ner_result"])
        elif para["feature"] == "autophrase":
            vocab_lst = utils.phrase_preprocess(para['autophrase_result'])

        vocab_lst =np.unique(vocab_lst)
        combining = False
        if combining == "True":
            combining = True
        
        clf = model.build_model(newsgroups_train_X, newsgroups_train_y,model = para['model'], vectorizing = para['vectorizing'], vocab_lst = vocab_lst, combining = combining)
        print('=========================================================')
        print("Evaluate on Validation Dataset")
        val_pred = clf.predict(newsgroups_val_X) 
        utils.evaluate(newsgroups_val_y, val_pred)
        print('=========================================================')
        
        print('=========================================================')
        print("Evaluate on Test Dataset")
        val_pred = clf.predict(newsgroups_test_X) 
        utils.evaluate(newsgroups_test_y, val_pred)
        print('=========================================================')

        filename = para["model_save"]
        with open(filename + 'model.pkl', 'wb') as file:
            pickle.dump(clf, file)


        
    
        
if __name__ == '__main__':
    main(sys.argv[1:])
