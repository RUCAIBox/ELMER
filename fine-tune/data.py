import codecs
import json
import torch
import random
import math
import pickle
import os
import numpy as np
from torch.utils.data import Dataset

BOS_ID, PAD_ID, EOS_ID, MASK_ID = 0, 1, 2, 50264


class S2SDataset(Dataset):
    """Dataset for sequence-to-sequence generative models, i.e., ELMER"""

    def __init__(self, data_dir, tokenizer, data_usage):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.data_usage = data_usage

        self.source_input_ids, self.labels = self.prepare_data()

    def __len__(self):
        return len(self.source_input_ids)

    def __getitem__(self, idx):
        return self.source_input_ids[idx], self.labels[idx]

    def prepare_data(self):
        """
        read corpus file
        """
        try:
            data = torch.load(os.path.join(self.data_dir, '{}.tar'.format(self.data_usage)))
            source_input_ids, labels = data["source_input_ids"], data["labels"]
        except FileNotFoundError:
            source_data, label_data = [], []

            source_file = os.path.join(self.data_dir, '{}.src'.format(self.data_usage))
            with codecs.open(source_file, "r") as fin:
                for line in fin:
                    source_data.append(line.strip())

            label_file = os.path.join(self.data_dir, '{}.tgt'.format(self.data_usage))
            with codecs.open(label_file, "r") as fin:
                for line in fin:
                    label_data.append(line.strip())

            assert len(source_data) == len(label_data)

            source_input_ids, labels = [], []
            for idx in range(len(source_data)):
                src_ids = self.tokenizer.encode(source_data[idx], add_special_tokens=True, truncation=True, max_length=512)
                lb_ids = self.tokenizer.encode(label_data[idx])
                source_input_ids.append(src_ids)
                labels.append(lb_ids)

            data = {"source_input_ids": source_input_ids, "labels": labels}
            torch.save(data, os.path.join(self.data_dir, '{}.tar'.format(self.data_usage)))

        return source_input_ids, labels
