import pandas as pd
from parse_text import spacy_txt
import json
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import word_tokenize
import numpy as np


def get_data(rxnorm_fp,mesh_fp,fp,QUERY):
    #read in ontology list
    
    rx=pd.read_csv(rxnorm_fp,usecols=['Preferred Label','Semantic type UMLS property'])
    mesh=pd.read_csv(mesh_fp,usecols=['Preferred Label','Semantic type UMLS property'])
    term = pd.concat([rx,mesh])
    term['Preferred Label'] = term['Preferred Label'].str.lower()
    term['Preferred Label'] = term['Preferred Label'].apply(lambda i :i.strip('(+)-').strip('(),.-{').strip().replace("'",'').replace("}",''))
    ontology = dict(zip(term['Preferred Label'],term['Semantic type UMLS property']))
    terms = list(ontology.keys())
    terms = sorted(terms)
    df,QUERY = parse_reddit(fp,QUERY)
    return terms,ontology,df,QUERY
def parse_reddit(fp,QUERY):
    with open(fp) as outfile:
        data = json.load(outfile)
    user=[]
    current_id=[]
    parent_id = []
    text=[]
    score=[]
    created = []
    url=[]

    is_submitter = []
    post_only = []
    def get_replies(comment,parent,first_reply):
        if len(comment['replies']) > 0:
            g = 0
            for n in comment['replies']:
                c = comment['replies'][n]
                user.append(n)
                if first_reply:
                    current = parent+'_reply_'+str(g)
                else:
                    current = parent+'.'+str(g)
                current_id.append(current)
                parent_id.append(parent)
                g+=1

                text.append(c['body'])
                score.append(c['score'])
                url.append(c['link'])
                created.append(c['created'])
                is_submitter.append(c['is_submitter'])
                post_only.append(np.nan)

                get_replies(c,current,False)
        else:
            return
    post_id = 0
    for i in data:
        if i =='https:':
            continue
        content = data[i]
        user.append(i)
        idd = 'post_'+str(post_id)
        current_id.append(idd)
        parent_id.append(np.nan)
        post_id +=1

        #append related content
        text.append(content['text'])
        score.append(content['score'])
        url.append(content['url'])
        created.append(content['created'])

        post_only.append({k:content[k] for k in ('author','flair','num_comments','title','subreddit') if k in content})
        is_submitter.append(np.nan)


        comment = content['comments']
        comment_id = 0
        if len(comment)>0:
            for c in comment:
                comments = comment[c]
                user.append(c)
                current_id.append(idd+'_comment_'+str(comment_id))
                parent_id.append(idd)

                text.append(comments['body'])
                score.append(comments['score'])
                url.append(comments['link'])
                created.append(comments['created'])
                is_submitter.append(comments['is_submitter'])
                post_only.append(np.nan)

                get_replies(comments,idd+'_comment_'+str(comment_id),True)
                comment_id +=1
    def clean_text(i):
        tokens = [word.strip() for word in nltk.word_tokenize(i)]
        stemmer = PorterStemmer()
        stems = [stemmer.stem(item) for item in tokens]
        return stems
    df=pd.DataFrame({'user':user,'parent_id':parent_id,'current_id':current_id,'text':text,'score':score,'created':created,"url":url,"is_submitter":is_submitter,"post_only":post_only})
    df.to_csv('parsed_reddit/scraper_output2_parsed.csv',index=False)
    df['clean_words'] = df['text'].apply(clean_text)
    df['clean_text'] = df['clean_words'].apply(lambda x : ' '.join(x))
    df.to_csv('parsed_reddit/'+QUERY+'_reddits.csv',index=False)
    return df,QUERY


