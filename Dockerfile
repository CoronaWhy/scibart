FROM python:3.7

MAINTAINER alex.walther "awalthermail@gmail.com"

WORKDIR /scibart

# install requirements
COPY requirements-train.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-train.txt

# create data directory
COPY data.zip .
RUN mkdir data && unzip data.zip -d data && rm data.zip

# copy code
RUN mkdir src
COPY src src/.

# run training
RUN python3 src/finetuning-bart.py
