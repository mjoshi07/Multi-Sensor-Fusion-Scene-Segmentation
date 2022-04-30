import torch.nn as nn

def cross_entropy_loss(model_output, gt):
    return nn.CrossEntropyLoss()(model_output, gt)
