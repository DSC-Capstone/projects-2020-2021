FROM ucsdets/scipy-ml-notebook:2020.2.9

USER root

# Install GATK
RUN pwd && \
    apt-get update && \
    apt-get install --yes default-jdk && \
    cd /opt && \
    wget -q https://github.com/broadinstitute/gatk/releases/download/4.1.4.1/gatk-4.1.4.1.zip && \
    unzip -q gatk-4.1.4.1.zip && \
    ln -s /opt/gatk-4.1.4.1/gatk /usr/bin/gatk && \
    rm gatk-4.1.4.1.zip && \
    cd /opt/gatk-4.1.4.1 && \
    ls -al  && \
    cd /home/jovyan

# install vcftools
RUN apt-get install --yes build-essential autoconf pkg-config zlib1g-dev && \
    cd /tmp && \
    wget -q -O vcftools.tar.gz https://github.com/vcftools/vcftools/releases/download/v0.1.16/vcftools-0.1.16.tar.gz && \
#    ls -al && \
    tar -xvf vcftools.tar.gz && \
    cd vcftools-0.1.16 && \
#    ls -al && \
    ./autogen.sh && \
    ./configure && \
    make && \
    make install && \
    rm -f /tmp/vcftools.tar.gz

# install samtools
RUN apt-get install --yes ncurses-dev libbz2-dev liblzma-dev && \
    cd /opt && \
    wget -q https://github.com/samtools/samtools/releases/download/1.10/samtools-1.10.tar.bz2 && \
    tar xvfj samtools-1.10.tar.bz2 && \
    cd samtools-1.10 && \
    ./configure && \
    make && \
    make install

# install bcftools
RUN apt-get install --yes ncurses-dev libbz2-dev liblzma-dev && \
    cd /opt && \
    wget -q https://github.com/samtools/bcftools/releases/download/1.10.2/bcftools-1.10.2.tar.bz2 && \
    tar xvfj bcftools-1.10.2.tar.bz2 && \
    cd bcftools-1.10.2 && \
    ./configure && \
    make && \
    make install

# install htslib
RUN apt-get install --yes ncurses-dev libbz2-dev liblzma-dev && \
    cd /opt && \
    wget -q https://github.com/samtools/htslib/releases/download/1.10.2/htslib-1.10.2.tar.bz2 && \
    tar xvfj htslib-1.10.2.tar.bz2 && \
    cd htslib-1.10.2 && \
    ./configure && \
    make && \
    make install

# Install TrimGalore and cutadapt
RUN wget https://github.com/FelixKrueger/TrimGalore/archive/0.6.6.zip -P /tmp/ && \
    unzip /tmp/0.6.6.zip && \
    rm /tmp/0.6.6.zip && \
    mv TrimGalore-0.6.6 /opt/

# path /opt/conda/bin/cutadapt
RUN python3 -m pip install --upgrade cutadapt

# FastQC
RUN wget http://www.bioinformatics.babraham.ac.uk/projects/fastqc/fastqc_v0.11.5.zip -P /tmp && \
    unzip /tmp/fastqc_v0.11.5.zip && \
    mv FastQC /opt/ && \
    rm -rf /tmp/fastqc_* && \
    chmod 777 /opt/FastQC/fastqc

# STAR
RUN wget https://github.com/alexdobin/STAR/archive/2.5.2b.zip -P /tmp && \
    unzip /tmp/2.5.2b.zip && \
    mv STAR-* /opt/ && \
    rm -rf /tmp/*.zip

# Picard
RUN wget http://downloads.sourceforge.net/project/picard/picard-tools/1.88/picard-tools-1.88.zip -P /tmp && \
    unzip /tmp/picard-tools-1.88.zip && \
    mv picard-tools-* /opt/ && \
    rm /tmp/picard-tools-1.88.zip

# SRA Tools
RUN wget https://ftp-trace.ncbi.nlm.nih.gov/sra/sdk/2.10.8/sratoolkit.2.10.8-centos_linux64.tar.gz -P /tmp && \
    tar xvf /tmp/sratoolkit* && \
    mv sratoolkit* /opt/ && \
    rm -rf /tmp/*.tar.gz

RUN wget https://github.com/pachterlab/kallisto/releases/download/v0.42.4/kallisto_linux-v0.42.4.tar.gz -P /tmp && \
    tar -xvf /tmp/kallisto_linux-v0.42.4.tar.gz && \
    mv kallisto_* /opt/ && \
    rm /tmp/kallisto_linux-v0.42.4.tar.gz

# HTSeq
RUN pip install HTSeq

# VarScan
RUN mkdir /opt/varscan && \
    wget http://downloads.sourceforge.net/project/varscan/VarScan.v2.3.6.jar -P /opt/varscan

# gtfToGenePred
RUN mkdir /opt/gtfToGenePred && \
    wget http://hgdownload.cse.ucsc.edu/admin/exe/linux.x86_64/gtfToGenePred -P /opt/gtfToGenePred

# STAR-Fusion
RUN wget https://github.com/STAR-Fusion/STAR-Fusion/releases/download/v0.8.0/STAR-Fusion_v0.8.FULL.tar.gz -P /tmp && \
    tar -xvf /tmp/STAR-Fusion_v0.8.FULL.tar.gz && \
    mv STAR-* /opt/

# JSplice
RUN mkdir /opt/JSplice && \
    git clone https://github.com/yannchristinat/jsplice.git /opt/JSplice && \
    cd /opt/JSplice && \
    python3 setup.py install

# RUN chmod -R 777 /opt
# Install BWA
#RUN conda install -c bioconda bwa=0.7.15 plink2

# Install PLINK2
#RUN conda install -c bioconda plink2

# Get R Packages
RUN R -e "install.packages('DESeq2')"
RUN R -e "install.packages('WGCNA')"


RUN rm -rf /opt/*.bz2 && \
    chmod -R +x /opt/*

COPY r-bio.yaml /tmp
RUN conda env create --file /tmp/r-bio.yaml && \
    rm -rf /opt/conda/bin/R /opt/conda/lib/R && \
    ln -s /opt/conda/envs/r-bio/bin/R /opt/conda/bin/R && \
    ln -s /opt/conda/envs/r-bio/lib/R /opt/conda/lib/R

USER $NB_UID
