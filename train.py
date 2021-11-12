import argparse
import os

import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
import math

from dataset.dataset import Dataset
from utils.load_settings import load_settings
from model.Base_Transformer import Transformer


def train(model, optimizer, scheduler, train_data, dev_data, batch_size, checkpoint, gpu):
    #loss_fn = nn.CrossEntropyLoss()

    step_cnt = 0
    best_model_weights = None

    for pointer in tqdm(range(0, len(train_data), batch_size), desc='training'):
        model.train()  # model was in eval mode in evaluate(); re-activate the train mode
        optimizer.zero_grad()  # clear gradients first
        torch.cuda.empty_cache()  # releases all unoccupied cached memory

        step_cnt += 1
        sent_pairs = []
        labels = []
        for i in range(pointer, pointer+batch_size):
            if i >= len(train_data):
                break
            sents = train_data[i].get_texts()
            if len(word_tokenize(' '.join(sents))) > 300:
                continue
            sent_pairs.append(sents)
            labels.append(train_data[i].get_label())
        logits, _ = model.ff(sent_pairs, checkpoint)
        if logits is None:
            continue
        true_labels = torch.LongTensor(labels)
        if gpu:
            true_labels = true_labels.to('cuda')
        loss = loss_fn(logits, true_labels)

        # back propagate
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # update weights
        optimizer.step()

        # update training rate
        scheduler.step()

        if step_cnt % 2000 == 0:
            """
            acc = evaluate(model, dev_data, checkpoint, mute=True)
            logging.info('==> step {} dev acc: {}'.format(step_cnt, acc))
            if acc > best_acc:
                best_acc = acc
                best_model_weights = copy.deepcopy(model.cpu().state_dict())
                model.to('cuda')
            """
    return best_model_weights


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
    total_steps = math.ceil(
        settings['network']['epochs'] * len(train_set) * 1. / settings['network']['batch_size'])
    warmup_steps = int(total_steps * settings['network']['warmup_percent'])
    scheduler = transformers.get_scheduler(
        optimizer, 'WarmupConstant', warmup_steps=warmup_steps, t_total=total_steps)

    best_model_dic = None

    for ep in range(settings['network']['epochs']):
        print(
            f"-------------------------- EPOCH {ep} --------------------------")
        model_dic = train()
        if model_dic is not None:
            best_model_dic = model_dic

    assert best_model_dic is not None


if __name__ == "__main__":
    main()
