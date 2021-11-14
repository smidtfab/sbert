import torch
from utils import reader
from torch.utils.data import Dataset
from typing import Dict, List, Tuple

from utils.map_labels import mapStrToInt

class CustomSentenceBatching:
    def __init__(self):
        pass

    def __call__(
        self, batch: List[Tuple[str, str, int]]
    ) -> Tuple[Dict[str, torch.LongTensor], Dict[str, torch.LongTensor], torch.LongTensor]:
        x1, x2, y = zip(*batch)
        x1, x2 = list(x1), list(x2)
        y = torch.LongTensor(y)
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
            tuple: tuple containing the two sentences and label
        """

        # Index the data
        item = self.train_df[index]
        
        return item[0], item[1], mapStrToInt(item[2])

    def __len__(self):
        """Get the length of the dataset

        Returns:
            int: lenght of the dataset
        """
        return len(self.train_df)
