import torch
import torch.nn as nn
from .Base_Transformer import Transformer
from .Pooling import Pooling
from .Classifier import Classifier
from transformers import AutoTokenizer, AutoModel
class SBERT(nn.Sequential):
    def __init__(self, transformer, pooling, classifier):
        """The SBERT model which is composed of the modules
        defined before

        Args:
            transformer (torch.nn.Module): the transfomer module
            pooling (torch.nn.Module): the pooling layer module
            classifier (torch.nn.Module): the classifier module
        """
        super().__init__(transformer, pooling, classifier)
        self.transformer = transformer
        self.pooling = pooling
        self.classifier = classifier
        self._do_classification = True # initialized as do classification (for train)
    
    @property
    def do_classification(self):
        """Whether to do classification

        Returns:
            bool: do classification or not
        """
        return self._do_classification

    @do_classification.setter
    def do_classification(self, value):
        """Setter for do_classification

        Args:
            value (bool): whether to put the model in classification mode
        """
        self._do_classification = value

    def forward(self, features_sent1, features_sent2):
        """Forward pass for the network. Here, all the components are
        utilized and we pass the features through their forwards.

        Args:
            features_sent1 (list, str): list or single str of the first sentences
            features_sent2 (list, str): list or single str of the second sentences

        Returns:
            torch.Tensor: either the class scores or the sentence embeddings (depending on do_classification)
        """
        # u pass
        tokenizer_output_sent1, transformer_output_sent1 = self.transformer(features_sent1)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_sent1 = self.pooling(transformer_output_sent1, tokenizer_output_sent1['attention_mask'])   

        # v pass
        tokenizer_output_sent2, transformer_output_sent2 = self.transformer(features_sent2)

        # Perform pooling. In this case, max pooling.
        sentence_embeddings_sent2 = self.pooling(transformer_output_sent2, tokenizer_output_sent2['attention_mask'])   

        if(self.do_classification):
            model_output = self.classifier(sentence_embeddings_sent1, sentence_embeddings_sent2)
        else:
            model_output = sentence_embeddings_sent1, sentence_embeddings_sent2

        return model_output