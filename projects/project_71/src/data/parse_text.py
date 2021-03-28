import spacy
import os
import ParsedObj
from collections import Counter
import pandas as pd


class spacy_txt():

    def read_file(self, dir_path):
        e = []
        n = []
        docs = []
        for filename in sorted(os.listdir(dir_path)):
            doc = ParsedObj.bare_document()
            file = pd.read_csv(dir_path+"/"+filename)
            texts = file['text'].values
            for p, post in enumerate(texts):
                sent = ParsedObj.bare_sentence()
                ent, noun, verb = self.process_sent(post)
                n.extend(noun)
                e.extend(ent)
                sent.noun.extend(noun)
                sent.verb.extend(verb)
                sent.ner.extend(ent)
                sent.sentence_id=p
                doc.sentences[i]= sent
            doc.path=dir_path+"/"+filename
            doc.doc_id=filename
            docs.append(doc)
        return docs, e, n
    def process_sent(self, str_text):
        nlp = spacy.load("en_core_web_sm")
        merge_pipe = nlp.create_pipe("merge_noun_chunks")
        nlp.add_pipe(merge_pipe)
        merge_pipe = nlp.create_pipe("merge_entities")
        nlp.add_pipe(merge_pipe)
        merge_pipe = nlp.create_pipe("merge_subtokens")
        nlp.add_pipe(merge_pipe)
        merge_pipe = nlp.create_pipe("sentencizer")
        nlp.add_pipe(merge_pipe)
        noun = []
        verb = []
        entity = []
        sentence = {}

        doc = nlp(str_text)
        for token in doc:
            if not token.is_stop:
                if token.pos_ in ['PROPN', 'NN', 'NOUN', 'NNP']:
                 print(token.text)
                 noun.append(token.text)
                if token.pos_ in ['VERB']:
                    verb.append(token.text)
        for ent in doc.ents:
            print(ent.text,  ent.label_)
            e = {}
            e[str(ent.text)]=ent.label_
            entity.append(e)

        return entity, noun, verb

    def __init__(self):
        print("gg")



