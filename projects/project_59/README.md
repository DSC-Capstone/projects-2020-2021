![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/rcgonzal/m2v-adversarial-hindroid)

# m2vDroid: Perturbation-resilient metapath-based Android Malware Detection
An extension of the [HinDroid malware detection system](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf), but using [metapath2vec](https://ericdongyx.github.io/metapath2vec/m2v.html) to encode apps in the Heterogeneous Information Network. We then hope to make the model resilient to adversarial ML like Android HIV. See our [blog post](https://rcgonzalez9061.github.io/m2v-adversarial-hindroid/) for more details on our methods.

## Setup and Usage
To recreate our results, please use our Docker image: `rcgonzal/m2v-adversarial-hindroid` and have access to a directory of Android apps decompiled into smali. (This can be done with the [APKtool](https://ibotpeaches.github.io/Apktool/) and [Smali](https://github.com/JesusFreke/smali) -- included with our Docker image)

Our project can be run using `python run.py [data] [analysis] [model]` with each tag corresponding to different workflows and being executed in the order shown.

The `data` flag will trigger our ETL workflow. It parses apps, constructs the HIN, and then generates `features.csv` using metapath2vec. It will read additional parameters from `config/etl-params/etl-params.json`. Note that including `data_source` will search for apps in that directory and subdirectories, creating `app_list.csv`. Otherwise, an `app_list.csv` can be specified by simply placing it within `outfolder`:

```json
{
    "outfolder": "Path where graph data will be saved",
    "parse_params": {
        "data_source": "Path to folder of decompliled apps, optional.",
        "nprocs": "Number of threads to use when parsing",
        "recompute": "Boolean, whether or not to reparse apps. Default, skips apps that exist in app data heap"   
    },
    "feature_params": {
        "walk_args": "Arguments for stellargraph.data.UniformRandomMetaPathWalk",
        "w2v_args": "Arguments for gensim.models.Word2Vec, excl. walks"
    }
}
```

The `analysis` flag will generate analysis on our data and any necessary plots, reading in additional parameters from `config/analysis-params/analysis-params.json`. `jobs` is a dictionary of jobs to be performed along with their parmeters. Currently the only available job is `plots`:

```json
{
    "data_path": "path to folder of data to load (akin to the outfolder in etl-params)",
    "jobs": {
        "plots": {
            "update_figs": "Boolean, whether or not to update figures in report and blog post",
            "no_labels": "Boolean, whether or not to include class labels on plots"
        }
    }
}
```

