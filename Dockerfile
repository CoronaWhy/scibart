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
COPY src .

# run training
RUN ls
RUN python3 finetuning-bart.py