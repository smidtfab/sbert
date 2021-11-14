import argparse
import os

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataset.dataset import Dataset, CustomSentenceBatching
from utils.load_settings import load_settings
from model.Base_Transformer import Transformer
from model.Pooling import Pooling
from model.Classifier import Classifier
from model.Sentence_Transformer import SBERT
from utils.eval import get_predicted_labels
from utils.map_labels import mapIntToStr

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

    assert(args.sent1 is not None and args.sent2 is not None)

    if(model.do_classification):
        # Do forward pass for sentences provided through args
        model_output  = model(args.sent1, args.sent2)
        
        # Get the predicted label
        print("Model Output:")
        print(model_output)  
        y_hat = get_predicted_labels(model_output).item()
        print(f"Predicted class label: {mapIntToStr(y_hat)}")
    else: 
        # Do forward pass for sentences provided through args
        model_output = model(args.sent1, args.sent2)
        sentence1_embedding, sentence2_embedding = model_output
        print(f"Shape of one sentence embedding: {sentence1_embedding.shape}")
        print("Sentence Embedding:")
        print(sentence2_embedding)

if __name__ == "__main__":
    main()
