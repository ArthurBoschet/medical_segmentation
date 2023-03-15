import torch.nn as nn
import torch

class Conv3DActivation(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1, 
                 activation=nn.ReLU
                 ):
        '''
        3D convolution followed by non linear activation
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
        '''
        super(Conv3DActivation, self).__init__()
        self.convolution = torch.nn.Conv3d(
                                    in_channels, 
                                    out_channels, 
                                    kernel_size, 
                                    stride=stride, 
                                    padding=padding, 
                                    dilation=dilation
                                )
        self.activation = activation()

    def forward(self, x):
        '''
        Parameters:
            x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
            x (torch.Tensor): (N,C_out,D,H,W) input size
        '''
        x = self.convolution(x)
        x = self.activation(x)
        return x
  

class Conv3DNormActivation(Conv3DActivation):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 dilation=1,  
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 ):
        '''
        3D convolution followed by normalization and then non linear activation
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
        '''
    
        super(Conv3DNormActivation, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, stride, 
            padding, 
            dilation, 
            activation=activation
            )
        self.normalization = normalization(out_channels)

    def forward(self, x):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size

        Returns:
        x (torch.Tensor): (N,C_out,D,H,W) input size
        '''
        x = self.convolution(x)
        x = self.normalization(x)
        x = self.activation(x)
        return x
    

class BaseConvBlock(nn.Module):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            ):
        super(BaseConvBlock, self).__init__()
        '''
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
        '''

        if normalization is None:
            self.base_block = lambda in_channels: Conv3DActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation,  
                                activation=activation 
                            )
        else:
            self.base_block = lambda in_channels: Conv3DNormActivation(
                                in_channels, 
                                out_channels, 
                                kernel_size, 
                                stride=stride, 
                                padding=padding, 
                                dilation=dilation,  
                                activation=activation, 
                                normalization=normalization,
                            )
            
    def forward(self, x):
        raise NotImplementedError

    

class DoubleConvBlock(BaseConvBlock):
    def __init__(
            self,
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            padding=0, 
            dilation=1,  
            activation=nn.ReLU, 
            normalization=nn.BatchNorm3d,
            ):
        super(DoubleConvBlock, self).__init__(
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=stride, 
            padding=padding, 
            dilation=dilation,  
            activation=activation, 
            normalization=normalization,
        )
        '''
        Combines 2 basic convolutional blocks into one.
        Parameters:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int, tuple or str, optional): Padding added to all six sides of the input. Default: 0
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
        '''
            
        self.conv_block_1 =  self.base_block(in_channels)
        self.conv_block_2 =  self.base_block(out_channels)

            
    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x
