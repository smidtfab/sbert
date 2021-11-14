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
            model_output = torch.cat(sentence_embeddings_sent1, sentence_embeddings_sent2)

        return model_output


if __name__ == "__main__":

    text_1 = "A person on a horse jumps over a broken down airplane."
    text_2 = "A person is training his horse for a competition."

    text_3 = "A person on a horse jumps over a broken down airplane."
    text_4 = "A person is at a diner, ordering an omelette."

    data = [[text_1, text_2], [text_3, text_4]]

    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    encoded_input = tokenizer(data, padding=True, truncation=True, return_tensors='pt')
    print(encoded_input)

    with torch.no_grad():
        transformer = Transformer('bert-base-cased')
        pooling = Pooling()
        classifier = Classifier(768, 3)
        model = SBERT(transformer, pooling, classifier)
        output = model(encoded_input)

    print(output)