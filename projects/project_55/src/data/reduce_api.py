import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class tfidf_reduce_api:
    def __init__(self, path, num_apis, output_path):
        self.num_apis = num_apis
        self.output_path = output_path
        self._load_data(path)
        self._run_tfidf_vec(num_apis)
        self._save_reduce_api_csv(output_path, num_apis)
        #self.malware_df
        #self.popular_df
        #self.random_df
        #self.corpus_lst

    def _load_data(self, path):

        # load malware, popular, and random csv 
        self.malware_df = pd.read_csv(f'{path}malware.csv').iloc[:, 1:]
        self.popular_df = pd.read_csv(f'{path}popular-apps.csv').iloc[:, 1:]
        self.random_df = pd.read_csv(f'{path}random-apps.csv').iloc[:, 1:]

        print("Data Loaded")
        
        df_lst = [self.malware_df, self.popular_df, self.random_df]

        def generate_corpus(df_lst):
            corpus_lst = []
            for df in df_lst:
                # combine api package and api name 
                df['api'] = 'pack' + df['package_id'].astype(str)+ 'api' + df['api_id'].astype(str)
                api_lst = df.groupby('app_id').agg(list)['api']
                for i in api_lst:
                    corpus_lst.append(' '.join(i))
            return corpus_lst

        self.corpus_lst = generate_corpus(df_lst)

        print("Corpus Created")

    def _run_tfidf_vec(self, num_apis):
        # initialize vectorizer
        tfIdfVectorizer=TfidfVectorizer(smooth_idf=True)

        # transform the corpus 
        tfIdf = tfIdfVectorizer.fit_transform(self.corpus_lst)

        # get the mean 
        df = pd.DataFrame(np.mean(tfIdf, axis=0).tolist()[0], index=tfIdfVectorizer.get_feature_names(), columns=['tfidf'])

        # sort by tfidf 
        self.df = df.sort_values('tfidf', ascending=False)

        # save the top #num of apis
        self.api_lst = self.df.head(num_apis).index.tolist()

        print("Top Api List Created")

    def _save_reduce_api_csv(self, output_path, num_apis):
        malware_top = self.malware_df[self.malware_df.api.isin(self.api_lst)]
        popular_top = self.popular_df[self.popular_df.api.isin(self.api_lst)]
        random_top = self.random_df[self.random_df.api.isin(self.api_lst)]

        malware_top.to_csv(f'{output_path}malware_tfidf_{num_apis}.csv')
        popular_top.to_csv(f'{output_path}popular_tfidf_{num_apis}.csv')
        random_top.to_csv(f'{output_path}random_tfidf_{num_apis}.csv')

        print("Result Saved")

def run_reduce_api(path, num_apis, output_path):
    tfidf_reduce_api(path, num_apis, output_path)






