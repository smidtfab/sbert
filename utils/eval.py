import torch

def get_predicted_labels(model_output): 
    softmaxed_outputs = torch.nn.functional.softmax(model_output)
    y_hat = torch.argmax(softmaxed_outputs, dim=1)
    return y_hat
    
def calculate_accuracy(y_hat, y):
    print(y_hat, y)
    return torch.sum(y_hat == y).item() / y_hat.shape[0]
