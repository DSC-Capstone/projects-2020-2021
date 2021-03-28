# referenced from https://github.com/ucsd-ets/datahub-example-notebook/blob/master/Dockerfile
# All requirements are directly taken from https://gitlab.com/dzhong1989/hvac-safety-control/-/blob/master/requirements.txt

# 1) choose base container
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

# 3) install packages (TODO: revise)
RUN pip install --no-cache-dir brickschema rdflib

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
