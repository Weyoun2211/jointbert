
from torch.utils.data import Dataset
import pandas as pd

from transformers import AutoTokenizer

from pdb import set_trace

class BertDataset(Dataset):

    def __init__(self, filename, tokenizer_name, max_length):

        # Store the contents of the file in a pandas dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self._prepare_data(filename, max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.data[index]

    def _prepare_data(self, filename, max_length):
        data = pd.read_pickle(filename, compression='gzip')
        data = data[['pair_id', 'sequence_left', 'sequence_right', 'label']]

        batch_encoding = self.tokenizer(data['sequence_left'].tolist(), data['sequence_right'].tolist(), truncation='longest_first', max_length=max_length)

        features = []
        labels = data['label'].tolist()
        self.pair_ids = data['pair_id'].tolist()

        self.dataframe = data[['pair_id', 'label']].copy()

        for i in range(len(data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            inputs['label'] = labels[i]

            features.append(inputs)
        return features

class BertDatasetJoint(Dataset):

    def __init__(self, filename, tokenizer_name, max_length):

        # Store the contents of the file in a pandas dataframe
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.data = self._prepare_data(filename, max_length)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def _prepare_data(self, filename, max_length):
        data = pd.read_pickle(filename, compression='gzip')
        data = data[['pair_id', 'sequence_left', 'sequence_right', 'label', 'label_multi1', 'label_multi2']]

        batch_encoding = self.tokenizer(data['sequence_left'].tolist(), data['sequence_right'].tolist(),
                                        truncation='longest_first', max_length=max_length)

        features = []
        labels = data['label'].tolist()
        labels_multi1 = data['label_multi1'].tolist()
        labels_multi2 = data['label_multi2'].tolist()
        self.pair_ids = data['pair_id'].tolist()

        self.dataframe = data[['pair_id', 'label', 'label_multi1', 'label_multi2']].copy()

        for i in range(len(data)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}
            inputs['label'] = labels[i]
            inputs['label_multi1'] = labels_multi1[i]
            inputs['label_multi2'] = labels_multi2[i]

            features.append(inputs)
        return features
