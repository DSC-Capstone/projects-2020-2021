import re
import pandas as pd
def update_entity_type(docs,similar_ls,terms,ontology,out_path):
    doc_indexs = {}
    for index,doc in enumerate(docs):
        doc_indexs[doc.doc_id] = index

    dic={'Entity_index':[],'Entity':[],'matched ontology':[],'jaccard similarity score':[],'ontology type':[]}
    for i in similar_ls:
        word_id = i[0]
        term_id = i[1]
        score = i[2]

        #get the sentence and word index of the entity
        doc_id = re.findall('.+(?=_sentence)',word_id)[0]
        doc_index = doc_indexs[doc_id]
        sent_index= int(re.findall('(?<=sentence_).+(?=_noun)',word_id)[0])
        noun_index=int(re.findall('(?<=noun_).+',word_id)[0])
        
        #get the entity 
        doc = docs[doc_index]
        sent = doc.sentences[sent_index]
        word = sent.noun[noun_index]
        
        types = []
        scores = []
        # get ontology type for entity
        onto = ontology[terms[term_id]]
        types.append(onto)
        scores.append(score)
        
        
        #check if the noun is already in ner list
        in_ner =False
        ner_score =1  
        #loop through list of entity dictionary for the sentence
        for entity in sent.ner:
            current = list(entity.keys())[0]
            if type(entity[current])!=str:
                    in_ner = True
            elif word==current.lower():
                types.append(entity[current])
                scores.append(ner_score)
                entity[current] = [types,scores]
                in_ner = True

        if not in_ner:
            #store type of entity to its location in the list
            sent.ner.append({word:[types,scores]})
            print(doc_index,sent_index)
        dic['Entity_index'].append(word_id)
        dic['Entity'].append(word)
        dic['matched ontology'].append(terms[term_id])
        dic['jaccard similarity score'].append(score)
        dic['ontology type'].append(onto)
        
    df = pd.DataFrame(dic)
    df = df.sort_values('jaccard similarity score',ascending=False)
    df.to_csv(out_path,index=False)