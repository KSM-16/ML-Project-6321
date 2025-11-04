'''Loss Prediction Module in PyTorch.

Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''
import torch
import torch.nn as nn 
import torch.nn.functional as F 

# Loss Prediction Loss
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    # assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()

    return loss

class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512],
                 interm_dim=128, model='resnet18'):
        super(LossNet, self).__init__()

        if model == 'resnet50':
            feature_sizes = [56, 28, 14, 7]
            num_channels = [256, 512, 1024, 2048]

        self.GAP = []
        for feature_size in feature_sizes:
            if model == 'resnet50':
                self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))
            else:
                self.GAP.append(nn.AvgPool2d(feature_size))
        self.GAP = nn.ModuleList(self.GAP)

        self.FC = []
        for num_channel in num_channels:
            self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        self.linear = nn.Linear(len(num_channels) * interm_dim, 1)

    def forward(self, features):
        outs = []
        for i in range(len(features)):
            out = self.GAP[i](features[i])
            out = out.view(out.size(0), -1)
            out = F.relu(self.FC[i](out))
            outs.append(out)

        out = self.linear(torch.cat(outs, 1))
        return out