# General Use Container
ARG BASE_CONTAINER=ubuntu:20.04
FROM $BASE_CONTAINER

LABEL maintainer="Najeem Kanishka"

USER root

# install python and related dependencies
RUN apt update -y
RUN apt upgrade -y

RUN apt install -y git vim
RUN apt install -y wget curl
RUN apt install -y language-pack-en
RUN DEBIAN_FRONTEND=noninteractive apt install -y unzip openjdk-8-jre-headless xvfb libxi6 libgconf-2-4
RUN apt install -y python3 python3-pip python3-scipy
RUN apt install -y mysql-client mysql-server libmysqlclient-dev

# these files need to be installed separately of requirements
# the installers error out when installed normally
RUN pip3 install --no-cache-dir Cython
RUN pip3 install --no-cache-dir html5lib
RUN pip3 install --no-cache-dir -U scikit-learn

# change the locale to suppport unicode characters
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV PYTHONIOENCODING=UTF-8

### The following lines are adapted from: https://nander.cc/using-selenium-within-a-docker-container

RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'
RUN apt-get -y update
RUN apt-get install -y google-chrome-stable
RUN apt-get install -yqq unzip
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/` \
    curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE \
    `/chromedriver_linux64.zip
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/
ENV DISPLAY=:99

### End citation

# finally, install from requirements
COPY /requirements-docker.txt /
RUN pip3 install --no-cache-dir -r /requirements-docker.txt

RUN echo "alias python=python3" >> ~/.bash_aliases

COPY /docker-init.sh /
RUN chmod 755 /docker-init.sh
CMD ["./docker-init.sh"]
