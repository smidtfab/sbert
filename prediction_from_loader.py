import argparse
import os
import time 

import torch
from torch.utils.data import DataLoader

from dataset.dataset import Dataset, CustomSentenceBatching
from utils.load_settings import load_settings
from model.Base_Transformer import Transformer
from model.Pooling import Pooling
from model.Classifier import Classifier
from model.Sentence_Transformer import SBERT
from utils.eval import get_predicted_labels, calculate_accuracy

def main():
    # Parse args that have been provided
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--settings_path', help='path to settings')
    parser.add_argument('--sent1', help='first sentence to encode')
    parser.add_argument('--sent2', help='second sentence to encode')
    args = parser.parse_args()

    # Set settings path (either provided through args, if not look in current dir)
    if args.settings_path is not None:
        path_settings = args.settings_path

    else:
        path_settings = os.path.join(
            os.path.dirname(__file__), 'settings.json')

    # Load settings from json
    settings = load_settings(path_settings)

    # Load the data
    collate_fn = CustomSentenceBatching()
    dev_set = Dataset(settings['data']['training_data_path'], partition_label='dev')
    test_loader = DataLoader(dev_set, batch_size=settings['network']['batch_size'], collate_fn=collate_fn)

    # Get BERT model
    bert = Transformer(model_name=settings['network']['architecture'], tokenizer_name=settings['network']['tokenizer_name'])
    pool = Pooling()
    classifier = Classifier(sent_embedding_dim=settings['network']['sent_embedding_dim'], num_classes=settings['network']['num_target_classes'])
    model = SBERT(bert, pool, classifier)

    # Load model weights
    model.load_state_dict(torch.load(settings['prediction']['path_model_pt']))
    model.eval()

    # Whether to return the class scores (True) or sentence embeddings as tensors (False)
    model.do_classification = settings['prediction']['do_classification']

    # Init lists for further calculations (i.e. accuracy)
    all_y_hats = torch.Tensor([])
    all_y = torch.Tensor([])
    for step, batch in enumerate(test_loader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(test_loader)))

            # Unpack this training batch from our dataloader. 
            # `batch` contains three objects:
            #   [0]: encoded dictionary with input_ids, attention_masks for first sentence
            #   [1]: encoded dictionary with input_ids, attention_masks for second sentence
            #   [2]: labels 
            sentences_encoded_dict = batch[0]
            sentences2_encoded_dict = batch[1]
            labels = batch[2]

            # Append for total accuracy calculation
            all_y = torch.cat((all_y, labels))

            # Perform a forward pass
            model_output = model(sentences_encoded_dict, sentences2_encoded_dict)
            print(model_output)

            if(model.do_classification):
                # Map classification class scores to labels
                y_hat = get_predicted_labels(model_output)
                all_y_hats = torch.cat((all_y_hats, y_hat))
                batch_accuracy = calculate_accuracy(y_hat, labels)
                print(f"Batch accuracy = {batch_accuracy}")

    total_accuracy = calculate_accuracy(all_y_hats, all_y)
    print(f"Total accuracy = {total_accuracy}")    

if __name__ == "__main__":
    main()
