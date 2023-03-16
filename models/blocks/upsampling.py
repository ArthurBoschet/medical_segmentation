import torch
import torch.nn as nn
import torch.nn.functional as F


class InterpolateUpsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        '''
        Upsample with interpolation
        Parameters:
            scale_factor (float or Tuple[float]): factor by which to up scale the tensor
            mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        '''
        super(InterpolateUpsample, self).__init__()
        self.mode = mode
        self.scale_factor = scale_factor

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N, C, *) where * can be 1D,2D or 3D
        '''
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x


