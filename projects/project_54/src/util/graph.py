"""
Graph creation utilities taken from https://github.com/alvinwan/neural-backed-decision-trees

Used to construct induced hierarchies as well as visualize them
"""
import networkx as nx
import json
import random
from nbdt.utils import DATASETS, METHODS, fwd
from networkx.readwrite.json_graph import node_link_data, node_link_graph
from sklearn.cluster import AgglomerativeClustering
from pathlib import Path
import nbdt.models as models
import torch
import argparse
import os

from dir_grab import *
from wn_utils import *

def generate_graph_fname(
    method,
    seed=0,
    branching_factor=2,
    no_prune=False,
    fname="",
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    **kwargs,
):
    if fname:
        return fname

    fname = f"graph-{method}"
    if method == "random":
        if seed != 0:
            fname += f"-seed{seed}"
    if method == "induced":
        assert (
            checkpoint or arch
        ), "Induced hierarchy needs either `arch` or `checkpoint`"
        if induced_linkage != "ward" and induced_linkage is not None:
            fname += f"-linkage{induced_linkage}"
        if induced_affinity != "euclidean" and induced_affinity is not None:
            fname += f"-affinity{induced_affinity}"
        if checkpoint:
            checkpoint_stem = Path(checkpoint).stem
            if checkpoint_stem.startswith("ckpt-") and checkpoint_stem.count("-") >= 2:
                checkpoint_suffix = "-".join(checkpoint_stem.split("-")[2:])
                checkpoint_fname = checkpoint_suffix.replace("-induced", "")
            else:
                checkpoint_fname = checkpoint_stem
        else:
            checkpoint_fname = arch
        fname += f"-{checkpoint_fname}"
    if method in ("random", "induced"):
        if branching_factor != 2:
            fname += f"-branch{branching_factor}"
    if no_prune:
        fname += "-noprune"
    return fname


def get_graph_path_from_args(
    dataset,
    method,
    seed=0,
    branching_factor=2,
    no_prune=False,
    fname="",
    induced_linkage="ward",
    induced_affinity="euclidean",
    checkpoint=None,
    arch=None,
    **kwargs,
):
    fname = generate_graph_fname(
        method=method,
        seed=seed,
        branching_factor=branching_factor,
        no_prune=no_prune,
        fname=fname,
        induced_linkage=induced_linkage,
        induced_affinity=induced_affinity,
        checkpoint=checkpoint,
        arch=arch,
    )
    
    directory = get_directory(dataset)
    path = os.path.join(directory, f"{fname}.json")
    return path

################
# INDUCED TREE #
################

# used to check if centers can be squeezed from fc weights
MODEL_FC_KEYS = (
    "fc.weight",
    "linear.weight",
    "module.linear.weight",
    "module.net.linear.weight",
    "output.weight",
    "module.output.weight",
    "output.fc.weight",
    "module.output.fc.weight",
    "classifier.weight",
    "model.last_layer.3.weight",
)

def build_induced_graph(
    wnids,
    checkpoint,
    model=None,
    linkage="ward",
    affinity="euclidean",
    branching_factor=2,
    dataset="CIFAR10",
    state_dict=None,
):
    '''
    Builds induced graph for induced hierarchy construction using wordnet ids, a CNNs state_dict, and 
    networkx
    '''
    num_classes = len(wnids)
    assert (
        state_dict
    ), "Need to specify `state_dict`."
    
    # build centers from state_dict of model
    centers = get_centers_from_state_dict(state_dict)
    
    assert num_classes == centers.size(0), (
        f"The model FC supports {centers.size(0)} classes. However, the dataset"
        f" {dataset} features {num_classes} classes. Try passing the "
        "`--dataset` with the right number of classes."
    )

    if centers.is_cuda:
        centers = centers.cpu()
    
    # use directed graph
    G = nx.DiGraph()

    # add leaves
    for wnid in wnids:
        G.add_node(wnid)
        set_node_label(G, wnid_to_synset(wnid))

    # add rest of tree
    clustering = AgglomerativeClustering(
        linkage=linkage, n_clusters=branching_factor, affinity=affinity,
    ).fit(centers)
    children = clustering.children_
    index_to_wnid = {}

    for index, pair in enumerate(map(tuple, children)):
        child_wnids = []
        child_synsets = []
        for child in pair:
            if child < num_classes:
                child_wnid = wnids[child]
            else:
                child_wnid = index_to_wnid[child - num_classes]
            child_wnids.append(child_wnid)
            child_synsets.append(wnid_to_synset(child_wnid))

        parent = get_wordnet_meaning(G, child_synsets)
        parent_wnid = synset_to_wnid(parent)
        G.add_node(parent_wnid)
        set_node_label(G, parent)
        index_to_wnid[index] = parent_wnid

        for child_wnid in child_wnids:
            G.add_edge(parent_wnid, child_wnid)

    assert len(list(get_roots(G))) == 1, list(get_roots(G))
    return G

def get_centers_from_state_dict(state_dict):
    fc = None
    for key in MODEL_FC_KEYS:
        if key in state_dict:
            fc = state_dict[key].squeeze()
            break
    if fc is not None:
        return fc.detach()

####################
# AUGMENTING GRAPH #
####################


def augment_graph(G, extra, allow_imaginary=False, seed=0, max_retries=10000):
    """Augment graph G with extra% more nodes.

    e.g., If G has 100 nodes and extra = 0.5, the final graph will have 150
    nodes.
    """
    n = len(G.nodes)
    n_extra = int(extra / 100.0 * n)
    random.seed(seed)

    n_imaginary = 0
    for i in range(n_extra):
        candidate, is_imaginary_synset, children = get_new_node(G)
        if not is_imaginary_synset or (is_imaginary_synset and allow_imaginary):
            add_node_to_graph(G, candidate, children)
            n_imaginary += is_imaginary_synset
            continue

        # now, must be imaginary synset AND not allowed
        if n_imaginary > 0:  # hit max retries before, not likely to find real
            return G, i, n_imaginary

        retries, is_imaginary_synset = 0, True
        while is_imaginary_synset:
            candidate, is_imaginary_synset, children = get_new_node(G)
            if retries > max_retries:
                print(f"Exceeded max retries ({max_retries})")
                return G, i, n_imaginary
        add_node_to_graph(G, candidate, children)

    return G, n_extra, n_imaginary


def set_node_label(G, synset):
    nx.set_node_attributes(G, {synset_to_wnid(synset): synset_to_name(synset)}, "label")


def set_random_node_label(G, i):
    nx.set_node_attributes(G, {i: ""}, "label")


def get_new_node(G):
    """Get new candidate node for the graph"""
    root = get_root(G)
    nodes = list(
        filter(lambda node: node is not root and not node.startswith("f"), G.nodes)
    )

    children = get_new_adjacency(G, nodes)
    synsets = [wnid_to_synset(wnid) for wnid in children]

    candidate = get_wordnet_meaning(G, synsets)
    is_fake = candidate.pos() == "f"
    return candidate, is_fake, children


def add_node_to_graph(G, candidate, children):
    root = get_root(G)

    wnid = synset_to_wnid(candidate)
    G.add_node(wnid)
    set_node_label(G, candidate)

    for child in children:
        G.add_edge(wnid, child)
    G.add_edge(root, wnid)


def get_new_adjacency(G, nodes):
    adjacency = set(tuple(adj) for adj in G.adj.values())
    children = next(iter(adjacency))

    while children in adjacency:
        k = random.randint(2, 4)
        children = tuple(random.sample(nodes, k=k))
    return children


def prune_single_successor_nodes(G):
    for node in G.nodes:
        if len(G.succ[node]) == 1:
            succ = list(G.succ[node])[0]
            G = nx.contracted_nodes(G, succ, node, self_loops=False)
    return G

def get_roots(G):
    for node in G.nodes:
        if len(G.pred[node]) == 0:
            yield node
            
####################
# WORDNET + GRAPHS #
###################

def get_wordnet_meaning(G, synsets):
    hypernyms = get_common_hypernyms(synsets)
    candidate = pick_unseen_hypernym(G, hypernyms) if hypernyms else None
    if candidate is None:
        return FakeSynset.create_from_offset(len(G.nodes))
    return candidate

def deepest_synset(synsets):
    return max(synsets, key=lambda synset: synset.max_depth())


def get_common_hypernyms(synsets):
    if any(synset.pos() == "f" for synset in synsets):
        return set()
    common_hypernyms = set(synsets[0].common_hypernyms(synsets[1]))
    for synset in synsets[2:]:
        common_hypernyms &= set(synsets[0].common_hypernyms(synset))
    return common_hypernyms

def pick_unseen_hypernym(G, common_hypernyms):
    assert len(common_hypernyms) > 0

    candidate = deepest_synset(common_hypernyms)
    wnid = synset_to_wnid(candidate)

    while common_hypernyms and wnid in G.nodes:
        common_hypernyms -= {candidate}
        if not common_hypernyms:
            return None

        candidate = deepest_synset(common_hypernyms)
        wnid = synset_to_wnid(candidate)
    return candidate
