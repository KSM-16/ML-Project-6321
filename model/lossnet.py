import torch
import torch.nn as nn 
import torch.nn.functional as F 


# -------------------------------------------------------
# Loss Prediction Loss (used for ranking uncertainty)
# -------------------------------------------------------
def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    """
    Computes pairwise ranking loss between predicted loss and true loss.
    Encourages correct ordering of sample difficulties.
    """

    # Ensure symmetrical pairing (input vs reversed input)
    assert input.shape == input.flip(0).shape

    # Pairwise differences (first half vs flipped second half)
    input = (input - input.flip(0))[:len(input)//2]
    target = (target - target.flip(0))[:len(target)//2]
    target = target.detach()  # stop gradients from flowing to target loss

    # Generate +1 / -1 labels based on sign of target difference
    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1

    # Margin ranking loss
    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)

    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)

    else:
        NotImplementedError()

    return loss


# -------------------------------------------------------
# Loss Prediction Network (LossNet)
# Predicts sample difficulty using intermediate features
# -------------------------------------------------------
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4],
                 num_channels=[64, 128, 256, 512],
                 interm_dim=128, model='resnet18'):
        """
        LossNet receives multi-layer backbone features and predicts
        a scalar loss value per sample.
        """
        super(LossNet, self).__init__()

        # Adjust feature sizes for ResNet50
        if model == 'resnet50':
            feature_sizes = [56, 28, 14, 7]
            num_channels = [256, 512, 1024, 2048]

        # Global average pooling layers for each feature map
        self.GAP = []
        for feature_size in feature_sizes:
            if model == 'resnet50':
                self.GAP.append(nn.AdaptiveAvgPool2d((1, 1)))  # dynamic pooling
            else:
                self.GAP.append(nn.AvgPool2d(feature_size))     # fixed pooling
        self.GAP = nn.ModuleList(self.GAP)

        # Fully connected projection for each feature map
        self.FC = []
        for num_channel in num_channels:
            self.FC.append(nn.Linear(num_channel, interm_dim))
        self.FC = nn.ModuleList(self.FC)

        # Final regression layer predicting loss value
        self.linear = nn.Linear(len(num_channels) * interm_dim, 1)

    def forward(self, features):
        """Forward pass: process each feature map and combine predictions."""
        outs = []

        for i in range(len(features)):
            out = self.GAP[i](features[i])      # pool
            out = out.view(out.size(0), -1)     # flatten
            out = F.relu(self.FC[i](out))       # projection + activation
            outs.append(out)

        # Concatenate multi-layer features and predict loss
        out = self.linear(torch.cat(outs, 1))
        return out
