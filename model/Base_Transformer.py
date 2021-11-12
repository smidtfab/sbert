import torch
from torch import nn
import tqdm
import numpy as np
#from preprocessing import prepare_bert_input


class Transformer(nn.Module):
    def __init__(self, model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.tokenizer = torch.hub.load(
            'huggingface/pytorch-transformers', 'tokenizer', model_name)
        self.transformer = torch.hub.load(
            'huggingface/pytorch-transformers', 'model', model_name)

    def forward(self, sent_pair_list, checkpoint=True, bs=None):
        all_probs = None

        if bs is None:
            bs = self.batch_size
            no_prog_bar = True
        else:
            no_prog_bar = False

        for batch_idx in tqdm(range(0, len(sent_pair_list), bs), disable=no_prog_bar, desc='evaluate'):
            probs = self.ff(
                sent_pair_list[batch_idx:batch_idx+bs], checkpoint)[1].data.cpu().numpy()
            if all_probs is None:
                all_probs = probs
            else:
                all_probs = np.append(all_probs, probs, axis=0)

        labels = []

        for pp in all_probs:
            ll = np.argmax(pp)
            if ll == 0:
                labels.append('contradiction')
            elif ll == 1:
                labels.append('entail')
            else:
                assert ll == 2
                labels.append('neutral')

        return labels, all_probs

    def ff(self, sent_pair_list):
        ids, types, masks = build_batch(
            self.tokenizer, sent_pair_list, self.model_name)
        if ids is None:
            return None
        ids_tensor = torch.tensor(ids)
        types_tensor = torch.tensor(types)
        masks_tensor = torch.tensor(masks)

        ids_tensor = ids_tensor.to('cpu')
        types_tensor = types_tensor.to('cpu')
        masks_tensor = masks_tensor.to('cpu')

        encoded_layers, _ = self.transformer(
            input_ids=ids_tensor, token_type_ids=types_tensor, attention_mask=masks_tensor, return_dict=False)

        #logits = self.nli_head(cls_vecs)
        #probs = self.sm(logits)

        print(encoded_layers.shape)

        # to reduce gpu memory usage
        # del ids_tensor
        # del types_tensor
        # del masks_tensor
        # torch.cuda.empty_cache() # releases all unoccupied cached memory

        return encoded_layers


def get_tokenized_input(tokenizer, sent1, sent2, model_type):

    tokenized_text = tokenizer.tokenize(sent1, sent2, add_special_tokens=True)
    indexed_tokens = tokenizer.encode(sent1, sent2, add_special_tokens=True)
    print(tokenized_text)
    print(indexed_tokens)
    assert len(tokenized_text) == len(indexed_tokens)

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
    return indexed_tokens, segments_ids


def build_batch(tokenizer, text_list, model_type):
    token_id_list = []
    segment_list = []
    attention_masks = []
    longest = -1

    for pair in text_list:
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
        output = model.ff(data)
    print(output)
