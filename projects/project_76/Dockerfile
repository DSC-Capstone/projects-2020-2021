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

RUN	apt-get install -y aria2
RUN	apt-get install -y nmap
RUN	apt-get install -y traceroute

# 3) install packages

RUN pip install --no-cache-dir yfinance==0.1.55
RUN pip install --no-cache-dir pandas-datareader==0.9.0
RUN pip install --no-cache-dir beautifulsoup4==4.9.3
RUN pip install --no-cache-dir numpy==1.14.3
RUN pip install --no-cache-dir scipy==1.1.0
RUN pip install --no-cache-dir networkx==2.1
RUN pip install --no-cache-dir scikit-learn==0.19.2
RUN pip install --no-cache-dir matplotlib==3.3.2

# 4) change back to notebook user
# COPY /run_jupyter.sh /
# USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
#
