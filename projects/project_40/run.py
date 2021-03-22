
import sys
import json


sys.path.insert(0, 'src/classifier')
sys.path.insert(0, 'src/hypothesis_test')
sys.path.insert(0, 'src/etl')

from classifier import build_classifier
from paired_t_tests import hypo_test_on_numerical
from chi_square_tests import chi_square_test
from clean import data_cleaning


def main(tragets):

    if 'build-classifier' in targets:
        with open('config/data_cleaning_param.json') as fh:
            clean_cfg = json.load(fh)
        with open('config/classifier_param.json') as fh:
            classifier_cfg = json.load(fh)

        x,y = data_cleaning(**clean_cfg)
        build_classifier(x,y,**classifier_cfg)

    if 'hypo-test' in targets:
        with open('config/hypo_test_param.json') as fh:
            hypo_cfg = json.load(fh)
        hypo_test_on_numerical(**hypo_cfg)

    if 'chi-square-test' in targets:
        with open('config/chi_square_param.json') as fh:
            chi_cfg = json.load(fh)
        chi_square_test(**chi_cfg)

    if 'test' in targets:
        with open('config/data_cleaning_param.json') as fh:
            clean_cfg = json.load(fh)
        with open('config/classifier_param.json') as fh:
            classifier_cfg = json.load(fh)

        x,y = data_cleaning(**clean_cfg)
        build_classifier(x,y,**classifier_cfg)

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)
