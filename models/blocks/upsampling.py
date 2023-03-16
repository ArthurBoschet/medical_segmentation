import torch
import torch.nn as nn
import torch.nn.functional as F

#torch.nn.functional.interpolate(
class InterpolateUpsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super(InterpolateUpsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


