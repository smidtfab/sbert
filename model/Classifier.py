import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, sent_embedding_dim, num_classes):
        """The classifier module used for training the NLI task

        Args:
            sent_embedding_dim (int): the sentence embedding dimension (here 768)
            num_classes (int): the number of classes (here 3)
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sent_embedding_dim = sent_embedding_dim
        self.num_classes = num_classes
        self.softmax_layer = nn.Linear(3 * self.sent_embedding_dim, self.num_classes).to(self.device)

    def forward(self, pooled_embeddings_sent1, pooled_embeddings_sent2):
        """Forward pass for the classifier module. Applies different 
        concats and finally the linear layer to produce class scores

        Args:
            pooled_embeddings_sent1 (torch.Tensor): the pooled embeddings for the first sentence (N x 768)
            pooled_embeddings_sent2 (torch.Tensor): the pooled embeddings for the second sentence (N x 768)

        Returns:
            torch.Tensor: the logit class scores (N x num_classes)
        """
        sent_embeddings_to_cat = []

        # u, v
        sent_embeddings_to_cat.append(pooled_embeddings_sent1)
        sent_embeddings_to_cat.append(pooled_embeddings_sent2)

        # |u - v|
        abs_diff_between_sent_embeddings = torch.abs(pooled_embeddings_sent1 - pooled_embeddings_sent2)
        sent_embeddings_to_cat.append(abs_diff_between_sent_embeddings)

        # (u, v, |u-v|)
        softmax_input = torch.cat(sent_embeddings_to_cat, 1)

        # do classification
        softmax_output = self.softmax_layer(softmax_input.to(self.device))
        
        return softmax_output
