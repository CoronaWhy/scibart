"""module to finetune BART with summary data in the repo"""

import argparse
import logging
import os
import sys

from fastai2.basics import Datasets, RandomSplitter, Learner, ranger
from transformers import BartTokenizer

import settings
import utils


sys.path.append('..')
logging.getLogger().setLevel(100)


def exp(args: argparse.Namespace):

    # load data
    print('loading datasets')
    datasets = args.datasets
    if args.datasets:
        datasets = args.datasets.split(',')
    data = utils.load_data(args.data_path, datasets)
    print(f'dataset has {data.shape[0]} examples')

    # split data into training, tests, and validation splits
    print('splitting datasets for training and testing')
    train_ds, _, _ = utils.split_datasets(data)

    # create tokenized dataset for training
    print('preparing dataloaders')
    tokenizer = BartTokenizer.from_pretrained('bart-large-cnn', add_prefix_space=True)
    x_tfms = [utils.DataTransform(tokenizer, column='text', max_seq_len=args.max_seq_len)]
    y_tfms = [utils.DataTransform(tokenizer, column='summary', max_seq_len=args.max_seq_len)]
    dss = Datasets(
        train_ds,
        tfms=[x_tfms, y_tfms],
        splits=RandomSplitter(valid_pct=settings.PERC_VALIDATION_SET)(range(train_ds.shape[0]))
    )
    data_loaders = dss.dataloaders(bs=args.batch_size, device=settings.DEVICE)

    # train
    print('training')
    learn = Learner(
        data_loaders,
        utils.FastaiWrapper(),
        loss_func=utils.SummarisationLoss(),
        opt_func=ranger,
        splitter=utils.bart_splitter
    )
    learn.freeze_to(-1)
    learn.fit_flat_cos(1, lr=1e-4)

    learn.freeze_to(-2)
    learn.dls.train.bs = args.batch_size//2
    learn.dls.valid.bs = args.batch_size//2
    learn.lr_find()
    learn.fit_flat_cos(2, lr=1e-5)

     # save trained weights and metrics
    print(f'saving training results to {settings.TRAINING_OUTPUT_DIR}')
    if not os.path.exists(settings.TRAINING_OUTPUT_DIR):
        os.makedirs(settings.TRAINING_OUTPUT_DIR)

    training_metrics = pd.DataFrame(learn.recorder.metrics)
    training_metrics.columns = learn.recorder.metric_names
    learn.recorder.metrics.to_csv(
        f'{settings.TRAINING_OUTPUT_DIR}/training_metrics.csv',
        index=False,
    )
    learn.export(f'{settings.TRAINING_OUTPUT_DIR}/fintuned_bart.pkl')
    print('training finished!')


if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(
        "BART Tuning", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    PARSER.add_argument("--batch-size", help="Choose batch size", type=int, default=4)
    PARSER.add_argument("--datasets", help="Choose datasets to include (default includes all)", default=None)
    PARSER.add_argument("--max-seq-len", help="Max sequence length", default=512)
    PARSER.add_argument("--data-path", help="Path to data directories", default='data')

    ARGS = PARSER.parse_args()
    exp(ARGS)