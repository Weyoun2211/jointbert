from torchvision import datasets, transforms
from base import BaseDataLoader

from dataset.datasets import BertDataset, BertDatasetJoint, BertDatasetJointShuffle
import pandas as pd

from data_loader.data_collators import DataCollatorWithPadding

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class BertDataLoader(BaseDataLoader):
    """
    DataLoader for BERT encoded sequences
    """

    def __init__(self, data_dir, batch_size, file, valid_file=None, valid_batch_size=None, shuffle=True,
                 validation_split=-1, num_workers=1, tokenizer_name='bert-base-uncased', max_length=None, mlm=False):
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.dataset = BertDataset(file, self.tokenizer_name, self.max_length)
        self.valid_batch_size = valid_batch_size
        self.mlm = mlm

        data_collator = DataCollatorWithPadding(
            tokenizer=self.dataset.tokenizer, mlm=self.mlm)
        if validation_split == -1:
            valid_ids = pd.read_csv(valid_file)
            self.valid_ids = valid_ids['pair_id'].tolist()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=data_collator)

class BertDataLoaderJoint(BaseDataLoader):
    """
    DataLoader for BERT encoded sequences
    """

    def __init__(self, data_dir, batch_size, file, valid_file=None, valid_batch_size=None, shuffle=True,
                 validation_split=-1, num_workers=1, tokenizer_name='bert-base-uncased', max_length=None, mlm=False):
        self.data_dir = data_dir
        self.max_length = max_length
        self.tokenizer_name = tokenizer_name
        self.dataset = BertDatasetJoint(file, self.tokenizer_name, self.max_length)
        self.valid_batch_size = valid_batch_size
        self.mlm = mlm

        data_collator = DataCollatorWithPadding(
            tokenizer=self.dataset.tokenizer, mlm=self.mlm)

        if validation_split == -1:
            valid_ids = pd.read_csv(valid_file)
            self.valid_ids = valid_ids['pair_id'].tolist()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=data_collator)