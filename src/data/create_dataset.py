"""script to create dataset"""

import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd


# data settings
DATASET_VERSION = '1.0.0'  # dataset version tag
DIR_RAW_DATA = 'raw_data'  # dir with downloaded and unzipped datasets
DIR_DATA = 'data'  # output dir
NCHUNKS = 40   # number of data frame splits and parquet output files

# find paths
paths = glob.glob(os.path.join(DIR_RAW_DATA, '*'), recursive=True)
paths = [Path(p) for p in paths]

# columns to keep from original datasets
cols2keep = {
    'SemanticScholarAbstractSectionSummaryDataSet': [
        'paperSection', 'Summary'
    ],
    'ArxivStructuredAbstractSectionalSummaries': ['paperSection', 'Summary'],
    'wikihow': ['text', 'overview'],
    'curation_corpus': ['article_content', 'summary'],
}

# create data directory
if not os.path.exists(DIR_DATA):
    os.makedirs(DIR_DATA)

# appends everything to the same dataframe. not great, but so far it's
# managable
data = []
for p in tqdm(paths):

    name = os.path.split(p)[-1]

    print(f"processing {name}")

    # read files
    files = p.iterdir()

    # wrangle curation corpus
    if 'curation_corpus' in name:
        parent_dir = list(files)[0].parent
        summaries = pd.read_csv(parent_dir / 'curation-corpus-base.csv')
        text = pd.read_csv(
            parent_dir / 'curation-corpus-base-with-articles.csv'
        )
        text = text[text.html != 'Exception']
        df = pd.merge(text, summaries, on='url')[[
            'article_content', 'summary']
        ]

    # read wikihow data
    elif 'wikihow' in name:
        df = pd.read_csv(list(files)[0])

    # read everything else
    else:
        dfs = [
            pd.read_parquet(f) for f in files
            if 'parquet' in str(f)
        ]
        df = pd.concat(dfs)

    df = df[cols2keep[name]]
    df.columns = ['text', 'summary']
    df['data_src'] = name
    data.append(df)

    print(f"{name} done")
data = pd.concat(data, ignore_index=True)

# split dataframe into smaller chunks
datas = np.array_split(data, NCHUNKS)

# save dataframe as parquet files
print(f"writing dataset ({NCHUNKS} files) to {DIR_DATA}{os.sep}")
for ix, d in enumerate(tqdm(datas)):
    filename = f'data{str(ix).zfill(2)}-{DATASET_VERSION}'
    d.to_parquet(
        os.path.join(DIR_DATA, f'{filename}.parquet.gzip'),
        compression='gzip',
    )
    print(f'{filename} done')
