# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
# ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

# scipy/machine learning (tensorflow)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN apt-get -y install htop

# CUDA Toolkit
RUN conda install -y cudatoolkit=10.1 cudnn nccl && \
    conda clean --all -f -y

# Torch
RUN pip install --no-cache-dir \
    torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 \
    -f https://download.pytorch.org/whl/torch_stable.html

# Other packages
RUN pip install --no-cache-dir networkx scipy==1.6.0 python-louvain fastai==1.0.57 opencv-python pyts

# COPY requirements.txt /tmp/
# RUN pip install --requirement /tmp/requirements.txt
# COPY . /tmp/

# 4) change back to notebook user
COPY /run_jupyter.sh /
RUN chmod 755 /run_jupyter.sh
USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
