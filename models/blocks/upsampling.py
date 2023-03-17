import torch.nn as nn
import torch.nn.functional as F

class UpScale(nn.Module):
    def __init__(self, scale_factor, in_channels):
        '''
        Upsample in 3d
        Parameters:
            scale_factor (int): factor by which to upsample the tensor
            in_channels (int): number of channels in the input
        '''
        super(UpScale, self).__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        

class InterpolateUpsample(UpScale):
    def __init__(self, scale_factor, in_channels, mode="nearest"):
        '''
        Upsample with interpolation
        Parameters:
            scale_factor (float or Tuple[float]): factor by which to up scale the tensor
            in_channels (int): number of channels in the input
            mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        '''
        super(InterpolateUpsample, self).__init__(scale_factor, in_channels)
        self.mode = mode

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
        super(TransposeConv3dUpsample, self).__init__(scale_factor, in_channels)
        self.transpose_conv = nn.ConvTranspose3d(
                                self.in_channels, 
                                self.in_channels, 
                                self.scale_factor, 
                                stride=self.scale_factor, 
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


