import torch.nn as nn

class MaxPool3dDownscale(nn.Module):
    def __init__(self, downscale_factor):
        '''
        Downsample with maxpooling in 3d
        Parameters:
            downscale_factor (int): factor by which to downscale the tensor
        '''
        super(MaxPool3dDownscale, self).__init__()
        self.maxpool = nn.MaxPool3d(downscale_factor, stride=downscale_factor)

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N, C, D, H, W)

        Returns:
            x (torch.Tensor): (N, C, D/scale, H/scale, W/scale)
        '''
        x = self.maxpool(x)
        return x