import torch
from torch import nn
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
class Transformer(nn.Module):
    def __init__(self, model_name, batch_size=16, device='cpu') -> None:
        super().__init__()
        self.model_name = model_name
        #self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.transformer = AutoModel.from_pretrained('bert-base-cased')
        self.batch_size = batch_size
        self.device = device

    def forward(self, sentence_pair_list):
        #tokenizer_output = self.tokenizer(sentence_pair_list, max_length=128, padding=True, truncation='longest_first', add_special_tokens=True, return_tensors='pt')
        #b_input_sent = sentence_pair_list['input_ids'].to(self.device)
        transformer_output = self.transformer(**sentence_pair_list)
        #print(transformer_output)
        return sentence_pair_list, transformer_output

if __name__ == "__main__":

    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    text_3 = "This is a test"
    text_4 = "Next sentence"
    data = [[text_1, text_2], [text_3, text_4]]
    with torch.no_grad():
        model = Transformer('bert-base-cased')
        output = model(data)
    print(output)
