FROM jupyter/scipy-notebook:latest

USER root

### Necessary packages for network-stats
#
# Installing libpcap for pcapy
RUN apt-get update && \
    apt-get install -y \
    libpcap-dev

WORKDIR /usr/local/src

# Installing pcapy
RUN wget "https://github.com/helpsystems/pcapy/archive/0.11.5.tar.gz" && \
    tar -xf 0.11.5.tar.gz && \
    rm 0.11.5.tar.gz
    
#RUN pip install graphviz

RUN cd pcapy*/ && \
    python3 setup.py install

# Installing impacket
RUN wget "https://github.com/SecureAuthCorp/impacket/releases/download/impacket_0_9_21/impacket-0.9.21.tar.gz" && \
    tar -xf impacket*.tar.gz && \
    rm impacket*.tar.gz

RUN cd impacket*/ && \
    pip3 install .

WORKDIR /
RUN wget https://raw.githubusercontent.com/ucsd-ets/datahub-base-notebook/master/scripts/run_jupyter.sh && \
    chmod +x run_jupyter.sh