import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, sent_embedding_dim, num_classes) -> None:
        super().__init__()
        self.sent_embedding_dim = sent_embedding_dim
        self.num_classes = num_classes
        self.softmax_layer = nn.Linear(3 * self.sent_embedding_dim, self.num_classes)

    def forward(self, pooled_embeddings_sent1, pooled_embeddings_sent2):
        # retrieve pooled sentence input from previous model layers
        #print(f"Pooled Emb 1 shape -> {pooled_embeddings_sent1.shape}")
        #print(f"Pooled Emb 2 shape -> {pooled_embeddings_sent2.shape}")
        sent_embeddings_to_cat = []

        # u, v
        sent_embeddings_to_cat.append(pooled_embeddings_sent1)
        sent_embeddings_to_cat.append(pooled_embeddings_sent2)

        # |u - v|
        abs_diff_between_sent_embeddings = torch.abs(pooled_embeddings_sent1 - pooled_embeddings_sent2)
        sent_embeddings_to_cat.append(abs_diff_between_sent_embeddings)

        # (u, v, |u-v|)
        softmax_input = torch.cat(sent_embeddings_to_cat, 1)
        #print(f"Softmax input shape -> {softmax_input.shape}")

        # do classification
        softmax_output = self.softmax_layer(softmax_input)
        #print(f"Softmax output shape -> {softmax_output.shape}")
        
        return softmax_output
