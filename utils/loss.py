"""Checkpoint-compatible focal and Dice losses used by the released heads."""

import torch
import torch.nn as nn

from torchvision.ops import sigmoid_focal_loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, target):
        assert inputs.size() == target.size(), "input and target must have the same shape"
        return sigmoid_focal_loss(inputs, target, self.alpha, self.gamma, reduction='mean')

class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.b_dice = BinaryDiceLoss(smooth)

    def forward(self, inputs, targets):
        assert inputs.size() == targets.size(), "inputs and targets must have the same shape"
        N = targets.size(1)
        if inputs.dim() == 3:
            loss = self.b_dice(inputs, targets)
        else:
            log_inputs = inputs.log_softmax(dim=1).exp()
            loss = 0
            for i in range(N):
                loss += self.b_dice(log_inputs[:, i, :, :], targets[:, i, :, :])
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        N = targets.size()[0]
        input_flat = inputs.view(N, -1)
        targets_flat = targets.view(N, -1)

        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + self.smooth) / (input_flat.sum(1) + targets_flat.sum(1) + self.smooth)
        loss = 1 - N_dice_eff.sum() / N
        return loss
