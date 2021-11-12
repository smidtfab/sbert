import argparse
import os

from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
import math

from dataset.dataset import Dataset
from utils.load_settings import load_settings
from model.Base_Transformer import Transformer


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, checkpoint, gpu):
    pass


def main():
    # Parse args that have been provided
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--settings_path', help='path to settings')
    args = parser.parse_args()

    # Set settings path (either provided through args, if not look in current dir)
    if args.settings_path is not None:
        path_settings = args.settings_path

    else:
        path_settings = os.path.join(
            os.path.dirname(__file__), 'settings.json')

    # Load settings from json
    settings = load_settings(path_settings)
    print(settings)

    # Load the data
    train_set = Dataset(settings['data']['training_data_path'])
    train_loader = DataLoader(
        train_set, batch_size=settings['network']['batch_size'], shuffle=True)
    print(len(train_loader))

    # Get BERT model
    model = Transformer(model_name=settings['network']['architecture'])
    optimizer = Adam(model.parameters(),
                     lr=settings['network']['learning_rate'])
    total_steps = math.ceil(settings['network']['epochs'] * len(train_set) * 1. / settings['network']['batch_size'])
    warmup_steps = int(total_steps * warmup_percent)
    scheduler = transformers.get_scheduler(
        optimizer, 'WarmupConstant', warmup_steps=warmup_steps, t_total=total_steps)


if __name__ == "__main__":
    main()
