# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

# scipy/machine learning (tensorflow)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN	apt-get install htop
RUN	apt-get install --yes aria2
RUN	apt-get install --yes nmap
RUN	apt-get install --yes traceroute

# 3) install packages
RUN conda install --yes geopandas 
RUN pip install --no-cache-dir babypandas networkx scipy python-louvain
RUN pip install wikiextractor
RUN pip3 install bixin
RUN pip install spanish_sentiment_analysis
RUN pip install -U textblob
RUN pip install --user -U nltk

# 4) change back to notebook user
COPY /run_jupyter.sh /
RUN chmod 755 /run_jupyter.sh
USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
