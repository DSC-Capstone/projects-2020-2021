# Politics on Wikipedia
This project is focused on detecting political controversy in online communities. We use a bag-of-words model and a party-embed model, trained on the ideological books corpus (Sim et al, 2013) as well as congressional record data (api.govinfo.gov), and attempt to generalize this to Wikipedia articles, validating it on edit comments which explicitly mention reverting bias.


## Usage

This code is intended to be run with the dockerfile vasyasha/pow_docker

It relies on data from the ideological books corpus (Sim et al., 2013) with sub-sentential annotations (Iyyer et al., 2014). To download this data please visit https://people.cs.umass.edu/~miyyer/ibc/index.html where you can send an email to the address in order to obtain the full dataset.

Once obtained, please extract the dataset to **/data/full_ibc/**

Once this is done, please alter the config in **/config/get_ibc_params** accordingly.

To run, in terminal type:
```
python run.py *target*
```

## Description of Contents

### `run.py`

Main driver for running the project. The targets and their functions are:
* `scrape_anames` : scrapes political article names
* `retrieve_anames` : obtains political articles
* `ibc` : downloads test IBC data
* `interpret_ibc` : runs partyembed model on IBC data
* `revision_xmls` : downloads XML files for nine political Wikipedia articles
* `partyembed` : runs Rheault and Cochrane model on current-page Wikipedia articles
* `partyembed_time` : runs Rheault and Cochrane model on Wikipedia edit histories
* `all` : Runs the whole pipeline.
* `test`: Runs the pipeline with pre-loaded test data.

### `config/`

* `get_ibc_params.json` : Input parameters for running the ibc target.

* `interpret_ibc_params.json` : Input parameters for running the interpret_ibc target.

### `notebooks/`

* `Partyembed+IBC_EDA.ipynb` : Jupyter notebook for the exploratory data analysis on Party_embed and IBC.

### `src/`

* `libcode.py` : Library code.

### `src/etl/`

* `bias.py` : Preliminary function for extracting bias from Rheault and Cochrane model.
* `get_anames.py` : Scrapes relevant article names.
* `get_atexts.py` : Scrapes article contents for the list gathered above.
* `get_ibc.py` : Downloads sample IBC data. For the full dataset, please see **Usage** above.
* `get_revision_xmls.py` : Downloads xml files using Wikipedia API for our time series analysis

### `src/models/`

* `difflib_bigrams.py` : Finds text difference between two article states
* `get_gns_scores.py` : Assigns scores to the article texts according to Gentzkow, Shapiro, Taddy 2019.
* `get_x2_scores.py` : Gets x2 scores based on the formula from Gentzkow and Shapiro 2010 from the IBC.
* `gns_histories.py` : Applies Gentzkow and Shapiro 2010 approach on edit histories
* `loadIBC.py` : This project uses code from (Sim et al., 2013) and (Iyyer et al., 2014). As this was written in a previous version of python, these updated versions replace downloads made during the building process.
* `partyembed_current_pages.py` : Applies partyembed model to get scores for the current pages of political Wikipedia articles
* `partyembed_ibc.py` : This file extracts from the partyembed .issue() function the ideological leanings of each word in each sentence of the ideological books corpus. After applying an aggregate function on this data, it writes this to a csv.
* `partyembed_revisions.py` : Applies partyembed model on edit histories to find change over time.
* `treeUtil.py` : This project uses code from (Sim et al., 2013) and (Iyyer et al., 2014). As this was written in a previous version of python, these updated versions replace downloads made during the building process.


## Sources

Papers Referenced
* https://siepr.stanford.edu/sites/default/files/publications/16-028.pdf

* https://www.cs.toronto.edu/~gh/2528/RheaultCochraneOct2018.pdf

Data
* https://people.cs.umass.edu/~miyyer/ibc/index.html

* https://data.stanford.edu/congress_text

* https://dumps.wikimedia.org

