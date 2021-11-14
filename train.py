import argparse
import os

import tqdm
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import transformers
import math
import time  

from dataset.dataset import Dataset, CustomSentenceBatching
from utils.load_settings import load_settings
from model.Base_Transformer import Transformer
from model.Pooling import Pooling
from model.Classifier import Classifier
from model.Sentence_Transformer import SBERT

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
    #print(settings)

    # Load the data
    collate_fn = CustomSentenceBatching(tokenizer_name=settings['network']['tokenizer_name'])
    train_set = Dataset(settings['data']['training_data_path'])
    train_loader = DataLoader(train_set, batch_size=settings['network']['batch_size'], collate_fn=collate_fn)
    dev_set = Dataset(settings['data']['training_data_path'], partition_label='dev')
    dev_loader = DataLoader(dev_set, batch_size=settings['network']['batch_size'], collate_fn=collate_fn)
    #print(next(iter(train_loader)))
    #print("#####################################train_set")
    #print(len(train_loader))

    # Get BERT model
    bert = Transformer(model_name=settings['network']['architecture'])
    pool = Pooling()
    classifier = Classifier(sent_embedding_dim=settings['network']['sent_embedding_dim'], num_classes=settings['network']['num_target_classes'])
    model = SBERT(bert, pool, classifier)
    #print(model)

    # Get optimizer and scheduler
    optimizer = Adam(model.parameters(),
                     lr=settings['network']['learning_rate'])
    total_steps = math.ceil(
        settings['network']['epochs'] * len(train_set) * 1. / settings['network']['batch_size'])
    warmup_steps = int(total_steps * settings['network']['warmup_percent'])
    scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)

    loss_fn = nn.CrossEntropyLoss()

    device = 'cpu' # TODO: in settings
    print("Starting training ...")

    best_train_loss = math.inf
    for ep in range(settings['network']['epochs']):
        print(
            f"--- EPOCH {ep} ---", flush=True)
        #model_dic = train(Sentence_Transformer, optimizer, scheduler, train_loader, settings['network']['batch_size'], )
        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0.0

        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = time.time() - t0
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_loader), elapsed))

            # Unpack this training batch from our dataloader. 
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the 
            # `to` method.
            #
            # `batch` contains three objects:
            #   [0]: encoded dictionary with input_ids, attention_masks for first sentence
            #   [1]: encoded dictionary with input_ids, attention_masks for second sentence
            #   [2]: labels 
            #print(batch)
            sentences_encoded_dict = batch[0]
            sentences2_encoded_dict = batch[1]
            labels = batch[2]

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because 
            # accumulating the gradients is "convenient while training RNNs". 
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()        

            # Perform a forward pass (evaluate the model on this training batch).
            output = model(sentences_encoded_dict, sentences2_encoded_dict)
            print(output)

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value 
            # from the tensor.
            loss = loss_fn(output, labels)
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        total_dev_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for step, batch in enumerate(dev_loader):
                        
            sentences_encoded_dict = batch[0]
            sentences2_encoded_dict = batch[1]
            labels = batch[2]

            # Perform a forward pass (evaluate the model on this training batch).
            output = model(sentences_encoded_dict, sentences2_encoded_dict)
            print(output)

            dev_loss = loss_fn(output, labels)
            total_dev_loss += dev_loss.item()   

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_loader)       
        avg_dev_loss = total_dev_loss / len(dev_loader)

        if(avg_train_loss < best_train_loss):
            model_save_name = 'best_model_weights'
            torch.save(model.state_dict(), os.path.join(os.curdir, model_save_name))
            best_train_loss = avg_train_loss
        
        # Measure how long this epoch took.
        training_time = time.time() - t0

        print("", flush=True)
        print("  Average training loss: {0:.2f}".format(avg_train_loss), flush=True)
        print("  Average dev loss: {0:.2f}".format(avg_dev_loss), flush=True)
        print("  Training epoch duration: {:}".format(training_time), flush=True)
        

if __name__ == "__main__":
    main()
