import sys
import os
import json

sys.path.insert(0, 'src')

from build_features import create_struct
from eda import run_model
from feature_extraction import matrix_a
from feature_extraction import matrix_b
from feature_extraction import matrix_p
from model import AAT
from model import ABAT
from model import APAT
from model import APBPAT
from model import build_svm
from node_2_vector import node2vec
from word_2_vector import word2vec
def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''
    if 'test' in targets:
        print('\n')
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh) 
        print('creating data structure...')
        data_dict = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[0]
        malware_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[1]
        benign_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[2]
        print('creating matrices...')
        a = matrix_a(data_dict)
        b = matrix_b(data_dict)
        p = matrix_p(data_dict)
        print('CREATING NODE 2 VECTOR EMBEDDINGS')
        node2vec()
        print('CREATING WORD 2 VECTOR EMBEDDINGS')
        word2vec()
        print('CREATING METAPATH 2 VECTOR EMBEDDINGS')
        #metapath2vec()
        #print('creating kernels...')
        #aat = AAT(a)
        #abat = ABAT(a, b)
        #apat = APAT(a, p)
        #apbptat = APBPAT(a,p,b)
        #print('running hindroid model...')
        #print('\n')
        #print('AAT accuracy - ' + str(build_svm(aat, 'aat', malware_seen, benign_seen)[0]))
        #print('AAT f1_score - ' + str(build_svm(aat, 'aat', malware_seen, benign_seen)[1]))
        #print('ABAT accuracy - ' + str(build_svm(abat,'abat', malware_seen, benign_seen)[0]))
        #print('ABAT f1_score - ' + str(build_svm(abat, 'abat', malware_seen, benign_seen)[1]))
        #print('APAT accuracy - ' + str(build_svm(apat, 'apat', malware_seen, benign_seen)[0]))
        #print('APAT f1_score - ' + str(build_svm(apat, 'apat', malware_seen, benign_seen)[1]))
        #print('APBPTAT accuracy - ' + str(build_svm(apbptat,'apbpat', malware_seen, benign_seen)[0]))
        #print('APBPTAT f1_score - ' + str(build_svm(apbptat,'apbpat', malware_seen, benign_seen)[1]))
        #print('\n')

    if 'all' in targets:
        with open('config/data-params.json') as fh:
            data_cfg = json.load(fh) 
        print('creating data structure...')
        data_dict = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[0]
        malware_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[1]
        benign_seen = create_struct(data_cfg['MALWARE_PATH'], data_cfg['BENIGN_PATH'])[2]
        print('creating matrices...')
        a = matrix_a(data_dict)
        b = matrix_b(data_dict)
        p = matrix_p(data_dict)
        print('RUNNING NODE2VEC')
        
        #print('creating kernels...')
        #aat = AAT(a)
        #abat = ABAT(a, b)
        #apat = APAT(a, p)
        #print('\n')
        #print('running hindroid model...')
        #print('AAT accuracy - ' + str(build_svm(aat, malware_seen, benign_seen)[0]))
        #print('AAT f1_score - ' + str(build_svm(aat, malware_seen, benign_seen)[1]))
        #print('ABAT accuracy - ' + str(build_svm(abat,'abat', malware_seen, benign_seen)[0]))
        #print('ABAT f1_score - ' + str(build_svm(abat, 'abat', malware_seen, benign_seen)[1]))
        #print('APAT accuracy - ' + str(build_svm(apat, 'apat', malware_seen, benign_seen)[0]))
        #print('APAT f1_score - ' + str(build_svm(apat, 'apat', malware_seen, benign_seen)[1]))
        

if __name__ == '__main__':
    # run via:
    # python main.py data features model
    targets = sys.argv[1:]
    main(targets)

