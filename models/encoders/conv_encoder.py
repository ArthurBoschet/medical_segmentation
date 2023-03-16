import torch.nn as nn

#personal modules
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock

class ConvEncoder(nn.Module):
    def __init__(self,
                 kernel_size=3,
                 num_blocks=3,
                 downscale_factor=2,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 dropout=0,
                 ):
        '''
        Convolutional encoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            kernel_size (int or tuple): Size of the convolving kernel must be an odd number
            num_blocks (int): number of convolutional blocks in the encoder
            downscale_factor (int): factor by which to downscale the image along depth, height and width
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
        '''

        assert kernel_size % 2 == 1, "kernel size should be an odd number (standard practice)"

        self.block_instanciator = lambda in_channels, out_channels: block_type(
                                                            in_channels, 
                                                            out_channels, 
                                                            kernel_size,  
                                                            padding=kernel_size//2,   
                                                            activation=activation, 
                                                            normalization=normalization,
                                                            dropout=dropout,
                                                        )
            

            

        

