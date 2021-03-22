FROM ucsdets/datahub-base-notebook:2020.2-stable

USER root

RUN sed -i 's:^path-exclude=/usr/share/man:#path-exclude=/usr/share/man:' \
        /etc/dpkg/dpkg.cfg.d/excludes

# install linux packages
RUN apt-get update && \
    apt-get install tk-dev \
                    tcl-dev \
                    cmake \
                    wget \
                    default-jdk \
                    libbz2-dev \
                    apt-utils \
                    gdebi-core \
                    dpkg-sig \
                    man \
                    man-db \
                    manpages-posix \
                    bwidget \
                    -y


# build conda environment with required r packages
COPY r-bio.yaml /tmp
RUN conda env create --file /tmp/r-bio.yaml

ENV RSTUDIO_PKG=rstudio-server-1.2.5042-amd64.deb
ENV RSTUDIO_URL=https://download2.rstudio.org/server/bionic/amd64/${RSTUDIO_PKG}
ENV PATH="${PATH}:/usr/lib/rstudio-server/bin"
ENV LD_LIBRARY_PATH="/usr/lib/R/lib:/lib:/usr/lib/x86_64-linux-gnu:/usr/lib/jvm/java-7-openjdk-amd64/jre/lib/amd64/server:/opt/conda/envs/r-bio/bin/R/lib"
ENV SHELL=/bin/bash
ENV R_LIB_SITE=/opt/conda/envs/r-bio/lib/R/library

# install RStudio
RUN ln -s /opt/conda/envs/r-bio/bin/R /usr/bin/R && \
    gpg --keyserver keys.gnupg.net --recv-keys 3F32EE77E331692F && \
    curl -L ${RSTUDIO_URL} > ${RSTUDIO_PKG} && \
    dpkg-sig --verify ${RSTUDIO_PKG} && \
    gdebi -n ${RSTUDIO_PKG} && \
    rm -f ${RSTUDIO_PKG} && \
    echo '/opt/conda/envs/r-bio/bin/R' > /etc/ld.so.conf.d/r.conf && /sbin/ldconfig -v && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm -f /usr/bin/R && \
    pip install jupyter-rsession-proxy && \
    mkdir -p /etc/rstudio && echo 'auth-minimum-user-id=100' >> /etc/rstudio/rserver.conf && \
    ( echo 'http_proxy=${http_proxy-http://web.ucsd.edu:3128}' ; echo 'https_proxy=${https_proxy-http://web.ucsd.edu:3128}' ) >> /opt/conda/envs/r-bio/etc/Renviron.site && \
    ( echo 'LD_PRELOAD=/opt/k8s-support/lib/libnss_wrapper.so'; echo 'NSS_WRAPPER_PASSWD=/tmp/passwd.wrap'; echo 'NSS_WRAPPER_GROUP=/tmp/group.wrap' ) >> /opt/conda/envs/r-bio/etc/Renviron.site

# linux hackery to remove paths to default R
RUN rm -rf /opt/conda/bin/R /opt/conda/lib/R && \
    ln -s /opt/conda/envs/r-bio/bin/R /opt/conda/bin/R

# create py-bio conda environment with required python packages
COPY py-bio.yaml /tmp
RUN conda env create --file /tmp/py-bio.yaml && \
    conda run -n py-bio /bin/bash -c "ipython kernel install --name=py-bio"


RUN conda install -c conda-forge bash_kernel

RUN yes | unminimize || echo "done"

USER $NB_USER
