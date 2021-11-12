import torch
from torch import nn
import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModel
class Transformer(nn.Module):
    def __init__(self, model_name, batch_size=16) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.transformer = AutoModel.from_pretrained('bert-base-cased')
        self.batch_size = batch_size

    def forward(self, sentence_pair_list):
        tokenizer_output = self.tokenizer(sentence_pair_list, padding=True, add_special_tokens=True, return_tensors='pt')
        transformer_output = self.transformer(**tokenizer_output)
        print(transformer_output)
        return tokenizer_output, transformer_output

    def calculate_embedding(self, sentence_pair_list):
        ids, types, masks = build_batch(
            self.tokenizer, sentence_pair_list, self.model_name)

        if ids is None:
            return None

        ids_tensor = torch.tensor(ids)
        types_tensor = torch.tensor(types)
        masks_tensor = torch.tensor(masks)

        ids_tensor = ids_tensor.to('cpu')
        types_tensor = types_tensor.to('cpu')
        masks_tensor = masks_tensor.to('cpu')

        encoded_layers, _ = self.transformer(
            input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor)

        print(type(encoded_layers))
        print(encoded_layers.shape)

        return encoded_layers, masks_tensor


def get_tokenized_input(tokenizer, sent1, sent2, model_type):
    tokenizer_output = tokenizer([sent1, sent2], padding=True, add_special_tokens=True, return_tensors='pt')
    print(f"tokenizer_output {tokenizer_output}")
    print(type(tokenizer_output['attention_mask']))

    """
    #tokenized_text = tokenizer.tokenize(sent1, sent2, add_special_tokens=True)
    #indexed_tokens = tokenizer.encode(sent1, sent2, add_special_tokens=True)
    #print(f"TOKENIZED TEXT {tokenized_text}")
    #print(indexed_tokens)
    #assert len(tokenized_text) == len(indexed_tokens)

    if len(tokenized_text) > 500:
        return None, None
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = []
    sep_flag = False
    for i in range(len(indexed_tokens)):
        if 'roberta' in model_type and tokenized_text[i] == '</s>' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif 'bert-' in model_type and tokenized_text[i] == '[SEP]' and not sep_flag:
            segments_ids.append(0)
            sep_flag = True
        elif sep_flag:
            segments_ids.append(1)
        else:
            segments_ids.append(0)
    """
    return tokenizer_output


def build_batch(tokenizer, text_list, model_type):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    for pair in text_list:
        print(f"pair -> {pair}")
        sent1, sent2 = pair
        ids, segs = get_tokenized_input(tokenizer, sent1, sent2, model_type)
        if ids is None or segs is None:
            continue
        token_id_list.append(ids)
        segment_list.append(segs)
        attention_masks.append([1] * len(ids))
        if len(ids) > longest:
            longest = len(ids)

    if len(token_id_list) == 0:
        return None, None, None

    # padding
    assert(len(token_id_list) == len(segment_list))

    for ii in range(len(token_id_list)):
        token_id_list[ii] += [0] * (longest-len(token_id_list[ii]))
        attention_masks[ii] += [0] * (longest-len(attention_masks[ii]))
        segment_list[ii] += [1] * (longest-len(segment_list[ii]))

    return token_id_list, segment_list, attention_masks


if __name__ == "__main__":
    tokenizer = torch.hub.load(
        'huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    # Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
    indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])

    model = torch.hub.load(
        'huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    with torch.no_grad():
        encoded_layers, x = model(
            tokens_tensor, token_type_ids=segments_tensors, return_dict=False)

    # print(encoded_layers)
    # print(x)
    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    text_3 = "This is a test"
    text_4 = "Next sentence"
    data = [[text_1, text_2], [text_3, text_4]]
    with torch.no_grad():
        model = Transformer('bert-base-cased')
        output = model(data)
    print(output)
