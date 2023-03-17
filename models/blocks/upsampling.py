import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class UpScale(nn.Module):
    def __init__(self, current_shape, target_shape, in_channels):
        '''
        Upsample in 3d
        Parameters:
            current_shape (Tuple[int, int, int, int, int]): current shape of the tensor (N, C, D, H, W)
            target_shape (Tuple[int, int, int, int, int]): target shape of the tensor
            in_channels (int): number of channels in the input
        '''
        super(UpScale, self).__init__()

        #only keep (D, H, W)
        self.current_shape = np.array(current_shape)[2:]
        self.target_shape = np.array(target_shape)[2:]

        #number of channels
        self.in_channels = in_channels

        #scaling factors
        self._scale_factor_calculate_float()
        self._scale_factor_calculate_int_padding()


    def _scale_factor_calculate_float(self):
        self.scale_factor_float = tuple(self.target_shape / self.current_shape)


    def _scale_factor_calculate_int_padding(self):
        self.scale_factor_int = tuple(self.target_shape // self.current_shape)
        self.output_padding = tuple(self.target_shape % self.current_shape)


        

class InterpolateUpsample(UpScale):
    def __init__(self, current_shape, target_shape, in_channels, mode="nearest"):
        '''
        Upsample with interpolation
        Parameters:
            current_shape (Tuple[int, int, int, int, int]): current shape of the tensor
            target_shape (Tuple[int, int, int, int, int]): target shape of the tensor
            in_channels (int): number of channels in the input
            mode (str): algorithm used for upsampling: 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'. Default: 'nearest'
        '''
        super(InterpolateUpsample, self).__init__(current_shape, target_shape, in_channels)
        self.mode = mode


    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N, C, *) where * can be 1D,2D or 3D

        Returns:
            x (torch.Tensor): (N, C, D*scale, H*scale, W*scale)
        '''
        x = F.interpolate(x, scale_factor=self.scale_factor_float, mode=self.mode)
        return x
    

class TransposeConv3dUpsample(UpScale):
    def __init__(self, current_shape, target_shape, in_channels):
        '''
        Upsample with 3d transpose convolution
        Parameters:
            current_shape (Tuple[int, int, int, int, int]): current shape of the tensor
            target_shape (Tuple[int, int, int, int, int]): target shape of the tensor
            in_channels (int): number of channels in the input
        '''
        super(TransposeConv3dUpsample, self).__init__(current_shape, target_shape, in_channels)
        self.transpose_conv = nn.ConvTranspose3d(
                                self.in_channels, 
                                self.in_channels, 
                                self.scale_factor_int, 
                                stride=self.scale_factor_int, 
                                padding=0, 
                                dilation=1,
                                output_padding=self.output_padding
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


