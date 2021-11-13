import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, sent_embedding_dim, num_classes) -> None:
        super().__init__()
        self.sent_embedding_dim = sent_embedding_dim
        self.num_classes = num_classes
        self.softmax_layer = nn.Linear(3 * self.sent_embedding_dim, self.num_classes)

    def forward(self, pooled_sent_embeddings):
        # retrieve pooled sentence input from previous model layers
        sent1_pooled_embedding = pooled_sent_embeddings[0]
        sent2_pooled_embedding = pooled_sent_embeddings[1]
        print(sent1_pooled_embedding.shape)
        sent_embeddings_to_cat = []

        # u, v
        sent_embeddings_to_cat.append(sent1_pooled_embedding)
        sent_embeddings_to_cat.append(sent2_pooled_embedding)

        # |u - v|
        abs_diff_between_sent_embeddings = torch.abs(sent1_pooled_embedding - sent2_pooled_embedding)
        sent_embeddings_to_cat.append(abs_diff_between_sent_embeddings)

        # (u, v, |u-v|)
        softmax_input = torch.cat(sent_embeddings_to_cat)
        print(softmax_input.shape)
        # do classification
        softmax_output = self.softmax_layer(softmax_input)
        print(softmax_output)
        return softmax_output
