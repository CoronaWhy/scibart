"""module with utility functions"""

from pathlib import Path
from tqdm import tqdm
from typing import List

from fastai2.basics import Transform, Module, params
from fastai2.text.all import TensorText
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer, BartForConditionalGeneration, BartConfig
import torch
from torch.nn import functional as F

import settings


class DataTransform(Transform):
    def __init__(self, tokenizer: PreTrainedTokenizer, column: str, max_seq_len: int):
        self.tokenizer = tokenizer
        self.column = column
        self.max_seq_len = max_seq_len

    def encodes(self, inp):
        tokenized = self.tokenizer.batch_encode_plus(
            [list(inp[self.column])],
            max_length=self.max_seq_len,
            pad_to_max_length=True,
            return_tensors='pt'
        )
        return TensorText(tokenized['input_ids']).squeeze()

    def decodes(self, encoded):
        decoded = [
            self.tokenizer.decode(
                o,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            ) for o in encoded
        ]
        return decoded


def load_hf_model(config, pretrained=True, path=None):
    if pretrained:
        if path:
            model = BartForConditionalGeneration.from_pretrained(
                "bart-large-cnn",
                state_dict=torch.load(path, map_location=torch.device(settings.DEVICE)),
                config=config
            )
        else:
            model = BartForConditionalGeneration.from_pretrained("bart-large-cnn", config=config)
    else:
        model = BartForConditionalGeneration()

    return model.to(settings.DEVICE)


class FastaiWrapper(Module):
    def __init__(self):
        self.config = BartConfig(vocab_size=50264, output_past=True)
        self.bart = load_hf_model(config=self.config, pretrained=True)

    def forward(self, x):
        output = self.bart(x)[0]
        return output


def load_data(path: str, datasets: List[str] = None) -> pd.DataFrame:
    files = list(Path(path).iterdir())
    data = pd.concat(
        pd.read_parquet(f) for f in tqdm(files)
    )
    if datasets:
        data = data[data.data_src.isin(datasets)]  # filter out datasets
    data.drop('data_src', 1, inplace=True)
    return data


def split_datasets(data):
    train_ds, test_ds = train_test_split(data, test_size=settings.PERC_VALIDATION_SET, random_state=42)
    valid_ds, test_ds = train_test_split(test_ds, test_size=0.5, random_state=42)

    return train_ds, test_ds, valid_ds


class SummarisationLoss(Module):
    def __init__(self):
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, output, target):
        x = F.log_softmax(output, dim=-1)
        norm = (target != 1).data.sum()
        return self.criterion(x.contiguous().view(-1, x.size(-1)), target.contiguous().view(-1)) / norm


def bart_splitter(model):
    return [
        params(model.bart.model.encoder),
        params(model.bart.model.decoder.embed_tokens),
        params(model.bart.model.decoder.embed_positions),
        params(model.bart.model.decoder.layers),
        params(model.bart.model.decoder.layernorm_embedding),
    ]
