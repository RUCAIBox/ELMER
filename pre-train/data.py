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

    def __init__(self, data_dir, epoch):
        self.data_dir = data_dir
        self.epoch = epoch
        self.source_input_ids, self.labels = self.prepare_data()

    def __len__(self):
        return len(self.source_input_ids)

    def __getitem__(self, idx):
        return self.source_input_ids[idx], self.labels[idx]

    def prepare_data(self):
        """
        read corpus file
        """

        data = torch.load(os.path.join(self.data_dir, '{}.tar'.format(self.epoch)))
        source_input_ids, labels = data["source_input_ids"], data["label_ids"]

        return source_input_ids, labels
