import torch
import torch.nn as nn
from .Base_Transformer import Transformer
from .Pooling import Pooling
from .Classifier import Classifier
from transformers import AutoTokenizer, AutoModel
class SBERT(nn.Sequential):
    def __init__(self, transformer, pooling, classifier) -> None:
        super().__init__(transformer, pooling, classifier)
        self.transformer = transformer
        self.pooling = pooling
        self.classifier = classifier
        self._do_classification = True # initialized as do classification (for train)
    
    @property
    def do_classification(self):
        return self._do_classification

    @do_classification.setter
    def do_classification(self, value):
        self._do_classification = value

    def forward(self, features_sent1, features_sent2):
        # u pass
        tokenizer_output_sent1, transformer_output_sent1 = self.transformer(features_sent1)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_sent1 = self.pooling(transformer_output_sent1, tokenizer_output_sent1['attention_mask'])   
        #print(f"Shape sent1 embedding _> {sentence_embeddings_sent1.shape}")

        # v pass
        tokenizer_output_sent2, transformer_output_sent2 = self.transformer(features_sent2)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_sent2 = self.pooling(transformer_output_sent2, tokenizer_output_sent2['attention_mask'])   
        #print(f"Shape sent2 embedding _> {sentence_embeddings_sent2.shape}")

        if(self.do_classification):
            model_output = self.classifier(sentence_embeddings_sent1, sentence_embeddings_sent2)
        else:
            model_output = sentence_embeddings_sent1, sentence_embeddings_sent2

        return model_output