"""
Utilities for NLTK WordNet synsets and WordNet IDs, as well as any other
misc functions for this project
"""
import sys
sys.path.insert(0, "../util")

import networkx as nx
import json
import random
from nltk.corpus import wordnet as wn
from pathlib import Path
import nbdt.models as models
import torch
import argparse
import os
import nltk

from dir_grab import *

##########
# SYNSET #
##########

class FakeSynset:
    '''
    Class to create FakeSynsets when wordnet ids are not found.
    '''
    def __init__(self, wnid):
        self.wnid = wnid

        assert isinstance(wnid, str)

    @staticmethod
    def create_from_offset(offset):
        return FakeSynset("f{:08d}".format(offset))

    def offset(self):
        return int(self.wnid[1:])

    def pos(self):
        return "f"

    def name(self):
        return "(generated)"

    def definition(self):
        return "(generated)"

def synset_to_wnid(synset):
    '''
    convert synset to wordnet id
    '''
    return f"{synset.pos()}{synset.offset():08d}"


def wnid_to_synset(wnid):
    '''
    convert wordnet id to synset
    '''
    offset = int(wnid[1:])
    pos = wnid[0]

    try:
        return wn.synset_from_pos_and_offset(wnid[0], offset)
    except:
        return FakeSynset(wnid)

def synset_to_name(synset):
    '''
    converts synset to its class name in human readable form
    '''
    return synset.name().split(".")[0]

def get_directory(dataset, root="./data/hierarchies"):
    return os.path.join(root, dataset)

def get_wnids_from_dataset(dataset, root="./data/wnids"):
    directory = os.path.join(root, dataset)
    return get_wnids(f"{directory}.txt")

def get_wnids(path_wnids):
    if not os.path.exists(path_wnids):
        parent = Path(Path(__file__).parent.absolute()).parent
        print(f"No such file or directory: {path_wnids}. Looking in {str(parent)}")
        path_wnids = parent / path_wnids
    with open(path_wnids) as f:
        wnids = [wnid.strip() for wnid in f.readlines()]
    return wnids

def wnid_to_name(wnid):
    return synset_to_name(wnid_to_synset(wnid))

