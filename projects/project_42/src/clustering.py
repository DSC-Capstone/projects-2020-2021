import os
import json
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from pandarallel import pandarallel
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from torch.utils.data import DataLoader
from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    SentencesDataset,
    losses,
)
from .plot_clusters import plot_all_clusters

spacy_en = spacy.load("en_core_web_sm")


def finetune_sbert(model, df, rep_sents, finetune_cfg):
    """Finetune the Sentence-BERT."""
    # setup
    train_size = finetune_cfg.get("train_size", 200000)
    sample_per_pair = finetune_cfg.get("sample_per_pair", 5)
    train_batch_size = finetune_cfg.get("train_batch_size", 32)
    epochs = finetune_cfg.get("epochs", 1)
    train = []
    n_sampled = 0
    cnts = [0, 0]  # [neg, pos]
    max_label_size = train_size // 2
    genres = df.genres.apply(set)

    with tqdm(total=train_size, position=0) as pbar:
        # sample sentence pairs
        while n_sampled < train_size:
            id1, id2 = np.random.randint(0, len(df), 2)
            label = int(bool(set.intersection(genres[id1], genres[id2])))

            if cnts[label] > max_label_size:
                continue

            sent_pairs = np.stack(
                np.meshgrid(rep_sents[id1], rep_sents[id2])
            ).T.reshape(-1, 2)
            if len(sent_pairs) <= sample_per_pair:
                samples = sent_pairs
            else:
                samples_idx = np.random.choice(
                    sent_pairs.shape[0], sample_per_pair, replace=False
                )
                samples = sent_pairs[samples_idx]

            inexp = lambda pair: InputExample(texts=list(pair), label=label)
            samples = list(map(inexp, samples))
            train.extend(samples)

            n_sampled += len(samples)
            cnts[label] += len(samples)
            pbar.update(len(samples))

        # run finetune
        train_ds = SentencesDataset(train, model)
        train_obj = (
            DataLoader(train_ds, shuffle=True, batch_size=train_batch_size),
            losses.ContrastiveLoss(model=model),
        )
        model.fit(
            train_objectives=[train_obj], epochs=epochs, warmup_steps=100
        )
        os.makedirs("model/clustering/sbert", exist_ok=True)
        model.save("model/clustering/sbert")


def pick_rep_sentences(row, n=5):
    """
    Pick sentences with highest average sublinear word tf-idfs
    as the representatives of the full text.
    """
    summary, phrases = row.summary, row.phrases

    spacy_doc = spacy_en(summary)
    sents = np.array([s.text for s in spacy_doc.sents])
    if len(sents) <= n:
        return sents
    if not phrases:
        return np.random.choice(sents, n, replace=False)

    cntv = CountVectorizer(vocabulary=phrases)
    tfidf = TfidfTransformer(sublinear_tf=True)
    ranks = (
        tfidf.fit_transform(cntv.fit_transform(sents)).toarray().sum(axis=1)
    )
    ranks /= np.array([len(s) for s in spacy_doc.sents])
    return sents[np.argpartition(-ranks, n)[:n]]


def embed_doc(rep_sents, model):
    """
    Acquire document embedding by averaging the embeddings
    of representative sentences of the document.
    """
    rep_embed = np.apply_along_axis(model.encode, 0, rep_sents)
    return np.mean(rep_embed, axis=0)


def calc_all_embeddings(df, model, cfg):
    """Calculate all document embeddings."""

    # pick representative sentences from documents
    print("[clustering] choosing representative sentences...")
    if cfg.rep_sents_path != "":
        rep_sents = np.load(cfg.rep_sents_path, allow_pickle=True)
    elif cfg.num_workers == 1:
        rep_sents = df.apply(pick_rep_sentences, axis=1).to_numpy()
    else:
        pandarallel.initialize(nb_workers=cfg.num_workers)
        rep_sents = df.parallel_apply(pick_rep_sentences, axis=1).to_numpy()
    np.save("model/clustering/rep_sents.npy", rep_sents, allow_pickle=True)

    # finetune
    if cfg.sbert_finetune:
        print(f"[clustering] finetuning the model...")
        finetune_sbert(model, df, rep_sents, cfg.sbert_finetune_cfg)

    # calculate document embeddings
    print("[clustering] calculating document embeddings...")
    if cfg.doc_embs_path != "":
        doc_embs = np.load(cfg.doc_embs_path, allow_pickle=False)
    else:
        rs_tqdm = tqdm(rep_sents, position=0)
        doc_embs = np.array([embed_doc(rs, model) for rs in rs_tqdm])
    np.save("model/clustering/doc_embs.npy", doc_embs, allow_pickle=False)
    return doc_embs


def run_gmm(doc_embs, num_clusters=200):
    """Run gaussian mixture."""
    print(f"[clustering] running GMM...")
    if len(doc_embs) < 50:
        pca_embs = doc_embs
    else:
        pca_embs = PCA(n_components=50).fit_transform(doc_embs)
    n_comp = min([num_clusters, len(doc_embs)])
    gmm = GaussianMixture(n_components=n_comp, verbose=2).fit(pca_embs)
    return gmm.predict(pca_embs)


def summarize_clusters(df, clusters, num_repft, repft_min_sup):
    """Summarize the clusters using defined heuristics."""
    print("[clustering] summarizing clusters...")
    out = []
    phrase_cnts = df.phrases.explode().value_counts()
    phrase_cnts = phrase_cnts[phrase_cnts > 100]
    genre_cnts = df.genres.explode().value_counts()
    genre_cnts = genre_cnts[genre_cnts > 100]

    for c in np.unique(clusters):
        docs = df[clusters == c]

        pcnt = docs.phrases.explode().value_counts().sort_values()
        pkws = (-pcnt / phrase_cnts).dropna().sort_values().index

        gcnt = docs.genres.explode().value_counts().sort_values()
        gkws = (-gcnt / genre_cnts).dropna().sort_values().index

        out.append(
            {
                "count": len(docs),
                "quality_phrases": list(pkws[:10]),
                "genres": list(gkws[:10]),
            }
        )

    with open("data/out/cluster_summaries.json", "w") as outf:
        json.dump(out, outf)


def dim_reduction(embs, num_workers, method="PCA"):
    """Reduce the dimension of embeddings for visualization."""
    dr = PCA(n_components=2)
    if isinstance(method, str) and method.upper() == "TSNE":
        dr = TSNE(n_components=2, n_jobs=num_workers, verbose=1)
    print(f"[clustering] running {dr.__class__.__name__}...")
    dr_embs = dr.fit_transform(embs)
    np.save("model/clustering/dr_embs.npy", dr_embs, allow_pickle=False)
    return dr_embs


def run_clustering(config):
    """Run the clustering pipeline."""
    # read config
    cfg = ClusteringConfig(config)

    # setup
    print("[clustering] setting up...")
    os.makedirs("model/clustering", exist_ok=True)
    df = pd.read_pickle("data/out/data.pkl")
    sbert = SentenceTransformer(cfg.sbert_base)

    # run steps
    doc_embs = calc_all_embeddings(df, sbert, cfg)
    clusters = run_gmm(doc_embs, cfg.num_clusters)
    summarize_clusters(df, clusters, cfg.num_repft, cfg.repft_min_sup)
    dr_embs = dim_reduction(
        doc_embs, cfg.num_workers, cfg.dim_reduction_method
    )
    plot_all_clusters(df, dr_embs)


class SimilarityQueryHandler:
    """Handle cosine similarity queries given document embeddings."""

    def __init__(self, embs):
        """Constructor."""
        # euclidean distance on unit vectors ~ cosine similarity
        self.embs = normalize(embs)
        self.nn = NearestNeighbors().fit(self.embs)

    def most_similar(self, idx, n):
        """Find indices of n most similar movies given a movie's index."""
        emb = [self.embs[idx]]
        return self.nn.kneighbors(emb, n, return_distance=False)[0]


class ClusteringConfig:
    """A "namespace" for config."""

    def __init__(self, config):
        self.num_workers = config.get("clu_num_workers", 1)
        self.rep_sents_path = config.get("clu_rep_sentences_path", "")
        self.doc_embs_path = config.get("clu_doc_embeddings_path", "")
        self.dim_reduction_method = config.get("clu_dim_reduction", "PCA")
        self.sbert_base = config.get("clu_sbert_base", "stsb-distilbert-base")
        self.sbert_finetune = config.get("clu_sbert_finetune", False)
        self.sbert_finetune_cfg = config.get("clu_sbert_finetune_config", {})
        self.num_clusters = config.get("clu_num_clusters", 50)
        self.num_repft = config.get("clu_num_rep_features", 10)
        self.repft_min_sup = config.get("clu_rep_features_min_support", 100)
