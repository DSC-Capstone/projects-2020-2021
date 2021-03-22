import spacy

def denpendence_parsing(df):
    nlp = spacy.load("en_core_web_sm")
    merge_pipe = nlp.create_pipe("merge_noun_chunks")
    nlp.add_pipe(merge_pipe)
    merge_pipe = nlp.create_pipe("merge_entities")
    nlp.add_pipe(merge_pipe)
    merge_pipe = nlp.create_pipe("merge_subtokens")
    nlp.add_pipe(merge_pipe)
    merge_pipe = nlp.create_pipe("sentencizer")
    nlp.add_pipe(merge_pipe)
    def parse_sent(s):
        doc=nlp(s)
        ls = []
        for token in doc:
            ls.append([token.text,token.pos_, token.dep_, token.head.text, token.head.pos_,
                [child for child in token.children]])
        return ls
    df['dep'] = df[df['is_emotion']&(~df['matched_drugs'].isna())]['text'].apply(parse_sent)
    return df