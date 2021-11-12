import torch
import torch.nn as nn
from Base_Transformer import Transformer
from Pooling import Pooling

class SBERT(nn.Sequential):
    def __init__(self, transformer, pooling) -> None:
        #self.pooling = pooling

        super().__init__(transformer, pooling)
        self.transformer = transformer
        self.pooling = pooling
        """
        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Use pytorch device: {}".format(device))

        self._target_device = torch.device(device)
        """

    def forward(self, features):
        tokenizer_output, transformer_output = self.transformer(features)
        print(transformer_output)
        # Perform pooling. In this case, max pooling.
        sentence_embeddings = self.pooling(transformer_output, tokenizer_output['attention_mask'])   
        print(sentence_embeddings.shape)
        return sentence_embeddings


if __name__ == "__main__":

    text_1 = "Who was Jim Henson ?"
    text_2 = "Jim Henson was a puppeteer"

    text_3 = "This is a test"
    text_4 = "Next sentence"
    data = [[text_1, text_2], [text_3, text_4]]
    with torch.no_grad():
        transformer = Transformer('bert-base-cased')
        pooling = Pooling()
        model = SBERT(transformer, pooling)
        output = model(data)
    print(output)