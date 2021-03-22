FROM openjdk:8

MAINTAINER Vincent Le <vil069@ucsd.edu>

USER root

RUN \   
    echo "===> install g++" && \
    apt-get update && apt-get install -y --force-yes g++

RUN \
    echo "===> install make, curl, perl" && \
    apt-get update && apt-get install -y --force-yes make curl perl

RUN apt-get update && apt-get install -y python3.7 && apt-get install -y python3-pip
RUN apt-get update && apt-get install -y git
RUN pip3 install jupyter
RUN pip3 install notebook
RUN pip3 install pandas
RUN pip3 install matplotlib
RUN pip3 install numpy
RUN pip3 install nltk
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install flask
RUN pip3 install pyspark
RUN rm -f /usr/bin/python && ln -s /usr/bin/python3 /usr/bin/python

COPY /run_jupyter.sh /
RUN chmod 755 /run_jupyter.sh
USER $NB_UID
