import re
import nltk
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

def text_encode(data_file, phrase_file, n_unigrams, threshhold, **kwargs):
    print()
    print('===================================================================')
    print(' => Text encoding...')
    print()

    merged_data = pd.read_csv(data_file)

    #Cleaned Event Feature

    def event_clean(text):
        result = re.sub('\n', '', text)
        result = re.sub('\t+', '\t', result)
        result = re.split('\t', result)

        if len(result) > 0:
            result = [s.lower() for s in result[1:]] # exclude the first item
            cleaned_result = []
            for s in result:
                if ';' in s:
                    for sub in s.split(';'):
                        cleaned_result.append(sub.strip())
                else:
                    cleaned_result.append(s.strip())
            return cleaned_result
        else:
            return ['Missing']

    cleaned_event = merged_data['event_type'].apply(event_clean)
    merged_data.insert(3, 'cleaned_event', cleaned_event)

    def event_clean_2(text):
        result = []
        for event in text:
            cleaned = event.replace('2.02', '').strip()
            if cleaned != '' and cleaned not in result:
                result.append(cleaned)
        return result

    merged_data['cleaned_event'] = merged_data['cleaned_event'].apply(event_clean_2)

    #Creating Target Variable

    def up_down_stay(price):
        if abs(price) < 1:
            return 'STAY'
        if price < 0:
            return 'DOWN'
        else:
            return 'UP'

    merged_data['target'] = merged_data['targe_price_change'].apply(up_down_stay)

    #Unigram Encoding

    def uni_encoding(data, category):
        word_count = {}
        lemmatizer = WordNetLemmatizer()
        temp = data.loc[data['target'] == category]
        for form in tqdm(temp['full_text']):
            cleaned_form = re.sub(r'\W',' ', form)
            cleaned_form = re.sub(r'\s+',' ', cleaned_form)
            cleaned_form = re.sub(r'\d','', cleaned_form)
            cleaned_form = cleaned_form.lower()
            tokens = nltk.word_tokenize(cleaned_form)
            for token in tokens:
                word = lemmatizer.lemmatize(token)
                if word not in word_count.keys():
                    word_count[word] = 1
                else:
                    word_count[word] += 1
        return word_count

    print('  => Tokenizing Data for 3 Classes...')
    print()

    up_dict = uni_encoding(merged_data.loc[merged_data['dataset'] == 'train'], 'UP')
    up_dict = {key:val for key, val in up_dict.items() if val > 10}

    down_dict = uni_encoding(merged_data.loc[merged_data['dataset'] == 'train'], 'DOWN')
    down_dict = {key:val for key, val in down_dict.items() if val > 10}

    stay_dict = uni_encoding(merged_data.loc[merged_data['dataset'] == 'train'], 'STAY')
    stay_dict = {key:val for key, val in stay_dict.items() if val > 10}

    all_word_count = {**up_dict, **stay_dict, **down_dict}

    #Compute PMI for each Class

    print()
    print('  => Computing PMI for 3 Classes...')
    print()

    def pmi_calc(all_words_dict, category_dict):
        total_freq = sum(all_words_dict.values())
        class_freq = sum(category_dict.values())
        pmi_dict = {}
        for token in tqdm(category_dict.keys()):
            p_x = all_words_dict[token] / total_freq
            p_x_class = category_dict[token] / class_freq
            pmi_dict[token] = np.log(p_x_class / p_x)
        return pmi_dict

    #Takes n Best Unigrams

    top_n = n_unigrams // 3

    up_pmi = pmi_calc(all_word_count, up_dict)
    up_pmi = {key: up_pmi[key] for key in sorted(up_pmi, key = up_pmi.get, reverse = True)[:top_n]}

    down_pmi = pmi_calc(all_word_count, down_dict)
    down_pmi = {key: down_pmi[key] for key in sorted(down_pmi, key = down_pmi.get, reverse = True)[:top_n]}

    stay_pmi = pmi_calc(all_word_count, stay_dict)
    stay_pmi = {key: stay_pmi[key] for key in sorted(stay_pmi, key = stay_pmi.get, reverse = True)[:top_n]}

    highest_pmi = highest_pmi = {**up_pmi, **down_pmi, **stay_pmi}
    unigram_features = pd.DataFrame(data = highest_pmi.keys(), columns = ['unigrams'])
    # unigram_features.to_csv('./data/model_unigrams.csv', index = False)

    print()
    print('  => Encoding Unigrams...')
    print()

    form_vectors = []
    for form in tqdm(merged_data['full_text']):
        cleaned_form = re.sub(r'\W',' ', form)
        cleaned_form = re.sub(r'\s+',' ', cleaned_form)
        cleaned_form = re.sub(r'\d','', cleaned_form)
        cleaned_form = cleaned_form.lower()
        tokens = nltk.word_tokenize(cleaned_form)
        temp = []
        for token in highest_pmi:
            if token in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        form_vectors.append(temp)

    merged_data['unigram_vec'] = form_vectors

    #Quality Phrase Encoding

    print()
    print('  => Encoding Quality Phrases...')
    print()

    quality_phrases = pd.read_csv(phrase_file, sep = '\t', header = None)

    def clean(text):
        return text.lower()

    quality_phrases['cleaned'] = quality_phrases[1].apply(clean)

    top_phrases = quality_phrases['cleaned'].loc[quality_phrases[0] > threshhold].copy()

    phrase_vectors = []
    for form in tqdm(merged_data['full_text']):
        cleaned_form = form.lower()
        temp = []
        for phrase in top_phrases:
            if phrase in cleaned_form:
                temp.append(1)
            else:
                temp.append(0)
        phrase_vectors.append(temp)

    merged_data['phrase_vec'] = phrase_vectors

    # Copied from train.py
    n_phrases = len(merged_data['unigram_vec'].values[0])
    def select_phrases(phrases, n_phrases):
        return phrases[:n_phrases]
    merged_data['top_phrases'] = merged_data['phrase_vec'].apply(lambda x: select_phrases(x, n_phrases))

    print()
    print(' => Done feature_encoding!')
    print()

    return merged_data, unigram_features
