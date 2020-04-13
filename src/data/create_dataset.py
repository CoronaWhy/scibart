"""script to create dataset"""

import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd



paths = glob.glob('../../raw_data/*/', recursive=True)
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
output_dir = Path('../../data')
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# appends everything to the same dataframe. not great, but so far it's
# managable
data = []
for p in paths:

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
nchunks = 40
datas = np.array_split(data, nchunks)

# save dataframe as parquet files
for ix, d in enumerate(tqdm(datas)):
    d.to_parquet(
        output_dir / f'data{str(ix).zfill(2)}.parquet.gzip',
        compression='gzip',
    )
