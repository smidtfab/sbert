import argparse
import os

from dataset.dataset import Dataset
from torch.utils.data import DataLoader
from utils.load_settings import load_settings


def main():
    # Parse args that have been provided    
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--settings_path', help='path to settings')
    args = parser.parse_args()

    # Set settings path (either provided through args, if not look in current dir)
    if args.settings_path is not None:
        path_settings = args.settings_path

    else:
        path_settings = os.path.join(os.path.dirname(__file__), 'settings.json')

    # Load settings from json
    settings = load_settings(path_settings)
    print(settings)

    # Load the data
    train_set = Dataset(settings['data']['training_data_path'])
    train_loader = DataLoader(train_set, batch_size = settings['network']['batch_size'], shuffle=True)
    print(len(train_loader))

    # Get BERT model 

if __name__ == "__main__":
    main()
