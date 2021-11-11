from utils import reader
from torch.utils.data import Dataset

class Dataset(Dataset):
    """
    The dataset used for training the Network. An element
    contains id, text, label
    """
    def __init__(self, training_tsv_path, transform=None):
        """Init for the custom dataset. Read the data.

        Args:
            training_tsv_path (string): the path to the tsv
            transform (torch.transform, optional): the transformers applied to the data. Defaults to None.
        """
        self.train_df = reader.read_tsv(training_tsv_path)  
        self.transform = transform

    def __getitem__(self, index):
        """Get an item at an index.

        Args:
            index (int): the index

        Returns:
            list: list containing the two sentences and label
        """
       
        # Get the data
        item = self.train_df[index]

        # Apply possible transformations
        if self.transform is not None:
            pass

        return item

    def __len__(self):
        """Get the length of the dataset

        Returns:
            int: lenght of the dataset
        """
        return len(self.train_df)