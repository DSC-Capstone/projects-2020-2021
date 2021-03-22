ARG BASE_IMAGE=ucsdets/datascience-notebook:2020.2-stable
FROM ${BASE_IMAGE}

# change to root to install packages
USER root

# Move to working directory /build
WORKDIR /build

COPY . .

# install packages
RUN pip install --no-cache-dir pymavlink
RUN pip install --no-cache-dir dronekit-sitl
RUN pip install --no-cache-dir mavproxy
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir bokeh
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir matplotlib
RUN pip install --no-cache-dir utm

# run test target as default behavior for container
CMD ["python", "run.py", "test"]
