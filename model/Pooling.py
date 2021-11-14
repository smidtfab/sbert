import torch
import torch.nn as nn
class Pooling(nn.Module):
    def __init__(self):
        """Pooling module to create fixed sized embeddings"""
        super().__init__()

    def forward(self, model_output, attention_mask):
        """Forward pass. Applies mean pooling to the token
        embeddings.

        Args:
            model_output (dict): the transformer output
            attention_mask (torch.Tensor): the attention mask from the tokenizer

        Returns:
            torch.Tensor: the pooled embeddings of fixed size (N x 768)
        """
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
