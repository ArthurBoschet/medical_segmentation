import torch.nn as nn
import torch.nn.functional as F

class UpScale(nn.Module):
    def __init__(self):
        super(UpScale, self).__init__()

class InterpolateUpsample(UpScale):
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

        Returns:
            x (torch.Tensor): (N, C, D*scale, H*scale, W*scale)
        '''
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        return x
    

class TransposeConv3dUpsample(UpScale):
    def __init__(self, scale_factor, in_channels):
        '''
        Upsample with 3d transpose convolution
        Parameters:
            scale_factor (int): factor by which to up scale the tensor
            in_channels (int): number of channels in the input
        '''
        super(TransposeConv3dUpsample, self).__init__()
        self.transpose_conv = nn.ConvTranspose3d(
                                in_channels, 
                                in_channels, 
                                scale_factor, 
                                stride=scale_factor, 
                                padding=0, 
                                dilation=1
                                )

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N, C, D, H, W)

        Returns:
            x (torch.Tensor): (N, C, D*scale, H*scale, W*scale)
        '''
        x = self.transpose_conv(x)
        return x


