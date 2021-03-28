FROM ucsdets/datascience-notebook:2020.2-stable

USER root


# Install python3, pip3
RUN apt-get update --fix-missing && \
    apt-get install -y git \
                       build-essential \
                       cmake \
                       vim \
                       wget \
                       unzip \
                       ffmpeg

# Installing CUDA 10.1
RUN conda install cudatoolkit=10.1 \
          cudnn \
          nccl \
          -y

# Upgrade pip
RUN pip install --upgrade pip

# Installing pip packages
RUN pip install --no-cache-dir numpy \
                               scipy \
                               pandas \
                               pyyaml \
                               notebook \
                               matplotlib \
                               seaborn \
                               scikit-image \
                               scikit-learn \
                               Pillow \
                               tensorboard cmake \
                               python-math \
                               opencv-python \
                               ffmpeg-python

# Installing torch
RUN pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Switching to user
USER jovyan

# Setting nvidia-smi
RUN ln -s /usr/local/nvidia/bin/nvidia-smi /opt/conda/bin/nvidia-smi

WORKDIR /tmp

# Downloading RTABMAP eval repo
RUN /bin/bash -c "git clone https://github.com/sisaha9/slamevaluations.git slameval"

#Setting up Detectron2
USER $NB_UID:$NB_GID
ENV PATH=${PATH}:/usr/local/nvidia/bin

RUN pip install 'git+https://github.com/facebookresearch/fvcore'
RUN git clone https://github.com/sisaha9/detectron2 detectron2_repo
RUN pip install -e detectron2_repo