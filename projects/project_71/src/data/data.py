import pandas as pd
from parse_text import spacy_txt

def get_data(rxnorm_fp,mesh_fp,text_path):
    #read in ontology list
    
    rx=pd.read_csv(rxnorm_fp,usecols=['Preferred Label','Semantic type UMLS property'])
    mesh=pd.read_csv(mesh_fp,usecols=['Preferred Label','Semantic type UMLS property'])
    term = pd.concat([rx,mesh])
    term['Preferred Label'] = term['Preferred Label'].str.lower()
    ontology = dict(zip(term['Preferred Label'],term['Semantic type UMLS property']))
    terms = list(ontology.keys())
    terms = sorted([i.strip('(+)-').strip('(),.-{').strip().replace("'",'').replace("}",'') for i in terms])
    
    s = spacy_txt()
    docs, v, n= s.read_file(text_path)
        
    return terms,docs,ontology

