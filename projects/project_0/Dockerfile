# 1) choose base container
# generally use the most recent tag


# Base image https://hub.docker.com/u/rocker/
FROM rocker/r-base:latest

USER root

COPY ./ ./
RUN python ./run.py test
USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]