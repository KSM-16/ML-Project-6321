import torch
from torch import nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


def model_parallel(model):
    model['backbone'] = torch.nn.DataParallel(model['backbone'])
    model['module'] = torch.nn.DataParallel(model['module'])
    torch.backends.cudnn.benchmark = True
    return model