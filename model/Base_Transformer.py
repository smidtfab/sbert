import torch
from torch import nn

if __name__ == "__main__":
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-cased')

    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    # Tokenized input with special tokens around it (for BERT: [CLS] at the beginning and [SEP] at the end)
    indexed_tokens = tokenizer.encode(text_1, text_2, add_special_tokens=True)

    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # Convert inputs to PyTorch tensors
    segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor([indexed_tokens])

    model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-cased')

    with torch.no_grad():
        encoded_layers, x = model(tokens_tensor, token_type_ids=segments_tensors, return_dict=False)

    print(encoded_layers)
    print(x)
