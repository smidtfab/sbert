import torch
from utils import reader
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

class CustomSentenceBatching:
    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def __call__(
        self, batch: List[Tuple[str, str, int]]
    ) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor], torch.LongTensor]:
        x1, x2, y = zip(*batch)
        x1, x2 = list(x1), list(x2)
        #print(f"x1 -> {x1}")
        #print(f"x2 -> {x2}")
        #sent_pairs = list(zip(x1, x2))
        # Zip the tensors to create sentence pairs i.e. [[sent1.1, sent1.2], [2.1, 2.2]]
        #sent_pairs = [list(pair) for pair in zip(x1, x2)]
        #print(f"sent_pairs -> {sent_pairs}")
        x1 = self.tokenizer(x1, max_length=128, padding=True, truncation='longest_first', add_special_tokens=True, return_tensors='pt')
        x2 = self.tokenizer(x2, max_length=128, padding=True, truncation='longest_first', add_special_tokens=True, return_tensors='pt')
        y = torch.LongTensor(y)
        #print(f"x1 -> {x1['input_ids'].shape}")
        #print(f"x2 -> {x2['input_ids'].shape}")
        #print(f"y -> {y.shape}")
        return x1, x2 , y
class Dataset(Dataset):
    """
    The dataset used for training the Network. An element
    contains id, text, label
    """

    def __init__(self, training_tsv_path, partition_label='train'):
        """Init for the custom dataset. Read the data.

        Args:
            training_tsv_path (string): the path to the tsv
            transform (torch.transform, optional): the transformers applied to the data. Defaults to None.
        """
        self.train_df = reader.read_tsv(training_tsv_path, partition_label)

    def __getitem__(self, index):
        """Get an item at an index.

        Args:
            index (int): the index

        Returns:
            list: list containing the two sentences and label
        """

        # Get the data
        item = self.train_df[index]

        #sent1, sent2 = item[0], item[1]
        sample = {}
        sample['sent1'] = item[0]
        sample['sent2'] = item[1]

        mapStrToInt = {"contradiction": 0, "entailment": 1, "neutral": 2}
        sample['label'] = mapStrToInt[item[2]]

        return sample['sent1'], sample['sent2'], sample['label']

    def __len__(self):
        """Get the length of the dataset

        Returns:
            int: lenght of the dataset
        """
        return len(self.train_df)
