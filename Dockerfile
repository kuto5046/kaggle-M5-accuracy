FROM ubuntu:18.04
 
# update
RUN apt-get -y update && apt-get install -y \
sudo \
wget \
vim \
git
 
#install anaconda3
WORKDIR /opt
# download anaconda package and install anaconda
# archive -> https://repo.continuum.io/archive/
RUN wget https://repo.continuum.io/archive/Anaconda3-2019.10-Linux-x86_64.sh && \
sh /opt/Anaconda3-2019.10-Linux-x86_64.sh -b -p /opt/anaconda3 && \
rm -f Anaconda3-2019.10-Linux-x86_64.sh

# set path
ENV PATH /opt/anaconda3/bin:$PATH
 
# update pip
RUN pip install --upgrade pip && \
    pip install lightgbm
    # pip install xonsh

RUN mkdir /work
# RUN touch ~/.xonshrc

# set shell