FROM python:3.7

MAINTAINER alex.walther "awalthermail@gmail.com"

WORKDIR /scibart

# install requirements
COPY requirements-train.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements-train.txt

# copy data
RUN mkdir data
COPY data data/.

# copy model
RUN mkdir src
COPY src src/.

# run training
RUN python3 src/model/finetuning-bart.py
