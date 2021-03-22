  
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable
FROM $BASE_CONTAINER
USER root



COPY . ./amr
#CMD ls ./PaperReplication

WORKDIR ./amr
RUN pip install -r ./requirements.txt
