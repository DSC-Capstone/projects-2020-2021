import pandas as pd
from operator import itemgetter
import json

def change_weight(unique_key, paper_ap_path, selected_domain, domain_ap_path, out_path, keyword_path):
    print("\n")
    print(">>>>>>>>>>>>>>>>>>>>>>>> Weighting quality scores <<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    print("  => Altering quality scores of phrases in the input document according to domain...")

    try:
        with open('config/domains_abb.json') as f:
            domains_name_dict = json.load(f)
#            add_key_path = 'data/out/selected_domain' + unique_key + '.txt'
            add_key_path = 'data/out/selected_domain.txt'
            with open(add_key_path) as f2:
                selected_domain = domains_name_dict[f2.readline()]
                domain_full_path = domain_ap_path + selected_domain + '/AutoPhrase.txt'
                print('Read domains phrases from '+ domain_full_path)
    except:
        domain_full_path = domain_ap_path + selected_domain + '/AutoPhrase.txt'
        print('Read domains phrases from '+ domain_full_path)

    data = pd.read_csv(paper_ap_path, sep="\t", header = None, names=["score", "phrase"])
    data_domain = pd.read_csv(domain_full_path, sep="\t", header = None, names=["score", "phrase"])
    domain_dict = data_domain.set_index('phrase').T.to_dict()
    phrases_dict = dict()

    domain_phrases = list(data_domain['phrase'])
    for i in range(data.shape[0]):
        phrase = data.loc[i, 'phrase']
        score = data.loc[i, 'score']
        if phrase in domain_phrases:
            phrases_dict[phrase] = score * domain_dict[phrase]['score']
            
    print("  => Saving results...")
    sorted_weight = sorted(phrases_dict.items(), key=itemgetter(1), reverse=True)
    result = pd.DataFrame(sorted_weight, columns= ['phrase', 'score'])
    result.to_csv(out_path)
    keywords = {'keywords': ', '.join(result['phrase'].iloc[:3])}
    with open(keyword_path, 'w') as fp:
        json.dump(keywords, fp)
    print(" => Done! Weighted result is saved as '" + out_path + "'")
    print("\n")
    return
