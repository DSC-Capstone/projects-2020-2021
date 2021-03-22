# DSC180BProject: Wikipedia’s Response to the COVID-19 Pandemic 


This is the Wikipedia project working on its performance on providing COVID-19 pandemic information. Most of our data generated can be seen using certain targets, but 
there are also some analysis we made through notebook and we will specified those notebooks in the notebooks seciton.

### Project Team Members:
- Yiheng Ye, yiy291@ucsd.edu
- Gabrielle Avila, ggavila@ucsd.edu
- Michael Lam, mel157@ucsd.edu

### Requirements:
- python 3.8
- pandas 1.1.0
- wordcloud 1.8.1
- wikipedia 1.4.0
- sklearn 0.24.1
- gensim 3.8.3
- nltk 3.5

### Code, Purpose, and Guideline:

- run.py: If target='data': Get top 1000 popular articles relating COVID-19 from Wikipedia. Get the pageview data for them in 2020.
          If target='eda': Get top10 article with top average daily pageview and plot their daily views
          If target='revision": Get revision history for important pages and doing analysis with LDA model on them.
          if target='word': Generates word cloud for Wikipedia, JHU, and WHO
          If target='test': Runs test program about data: getting pageview on the test data and eda, getting revision data and doing LDA model on them, and 
          generating word cloud.
- elt.py: the library for the data pipeline, see the documentation for detailed functions of every function writtened. Basically
          these functions are used to fulfill the job done in run.py.
- eda.py: the library for doing eda on data.
- revision.py: the library for analysis revision data
- word.py: the library for creating wordcloud
- config/data-params.json: it stores the links of the source data as well as the output path for raw data.
- code in src/data: the source code to fulfill the functions about processing data. The current usable files are get_data.py(getting top1000 articles'
  basic information) and get_apipageview.py(getting pageview from given article information csvs)

### Notebooks
The notebook file is primary serving as our original test base for code development. Additionally, it also has a notebook called Project EDA Single Webpage.ipynb which we investigate "COVID-19 pandemic data" page deeply.

There is also another notebook called "Word Clouds.ipynb" which produces word clouds on Wikipedia Coronavirus page, JHU page, and WHO page.

The "top_model.ipynb" generated LDA model for the LDA model on article 'Coronavirus", and this model needed to be open in a notebook to get visualization.

## Responsibilities:
- Yiheng Ye set up the structure of the project and the structure of run.py. He also wrote get_data.py and get_apipageview.py and put them into the etl.py. He also 
  wrote eda.py and eda_pageview.py
- Gabrielle Avila constructed our report and made deep analysis into the "COVID-19 pandemic data" page. She also made the "Word Clouds.ipynb"
- Michael Lam made LDA model analysis on the page "Coronavirus" and put them into the notebook "top_model.ipynb".