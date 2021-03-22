import sys
import os
import re
import json
# from utils import data_utils
####################################

class PartySet:
    
    """
    Iterator for Political Party's tweets
    """
    
    def __init__(self, path_to_party_tweets):
        self.data_folder = path_to_party_tweets
        self.filenames = os.listdir(self.data_folder)
        self.filepaths = [os.path.join(self.data_folder, fn) for fn in self.filenames]
        self.representatives = [re.findall(r'^(.*?)\_tweets\.jsonl', fn)[0] for fn in self.filenames]
        
    def text_iter(self):
        for file in self.filepaths:
            with open(file) as f:
                for line in f:
                    tweet = json.loads(line)
                    text = tweet['full_text']
#                     text = data_utils.clean_text(text)
                    yield(text)
    
    def get_all_text(self):
        all_text = []
        gen = self.text_iter()
        while True:
            try:
                text = next(gen)
                all_text.append(text)
            except StopIteration:
                break
        return all_text
