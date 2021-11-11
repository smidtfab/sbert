from ..utils import reader
from torch.utils.data import Dataset

class Dataset(Dataset):
    """
    The dataset used for training the Network. An element
    contains id, text, label
    """
    def __init__(self, training_tsv_path=None, transform=None):

        self.train_df = reader.read_tsv(training_tsv_path)  
        self.transform = transform

    def __getitem__(self, index):
       
        # Get the data
        item = self.train_df[index]

        # Apply possible transformations
        if self.transform is not None:
            pass

        return item

    def __len__(self):
        return len(self.train_df)