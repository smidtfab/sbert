import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
class Transformer(nn.Module):
    def __init__(self, model_name, tokenizer_name, batch_size=16):
        """Init the base transfomer model

        Args:
            model_name (str): the model name. Used to get the pretrained model (here bert)
            tokenizer_name (str): the tokenizer name. Used to get the tokenizer (here bert)
            batch_size (int, optional): the batch size. Defaults to 16.
        """
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.transformer = AutoModel.from_pretrained(model_name).to(self.device)
        self.batch_size = batch_size

    def forward(self, sentence_list):
        """Base Transformer in the siamese network. Used
        for tokenizing and transforming the input using BERT.   

        Args:
            sentence_list (list,str): a single sentence or list of sentences  

        Returns:
            [dict, torch.Tensor]: the dictionary from the tokenizer including input_ids, 
            attention_masks and the transformer output as a tensor
        """
        tokenizer_output = self.tokenizer(sentence_list, max_length=128, padding=True, truncation='longest_first', add_special_tokens=True, return_tensors='pt')
        transformer_output = self.transformer(**tokenizer_output.to(self.device))
        return tokenizer_output, transformer_output