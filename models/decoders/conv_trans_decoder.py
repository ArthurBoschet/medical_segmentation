import torch
import torch.nn as nn
import numpy as np

#personal modules
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from blocks.vision_multihead_attention import VisionMultiheadAttention
from blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample
from decoders.conv_trans_decoder import ConvDecoder
from utils.conv_utils import conv3d_output_dim


class ConvTransDecoder(ConvDecoder):
    def __init__(self,
                 encoder_shapes,
                 num_channels_list,
                 kernel_size=3,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 upsampling=TransposeConv3dUpsample,
                 skip_mode='append',
                 dropout=0,
                 ):
        '''
        Convolutional decoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            encoder_shapes (list): list of shapes (N,C,D,H,W) coming from the encoder in the order [skip1, skip2, ..., output]
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            upsampling (blocks.conv.downsampling.Downscale): upsampling scheme
            skip_mode (str): one of 'append' | 'add' refers to how the skip connection is added back to the decoder path
            dropout (float): dropout added to the layer
        '''
        super(ConvDecoder, self).__init__(
                                        encoder_shapes,
                                        num_channels_list,
                                        kernel_size=kernel_size,
                                        activation=activation, 
                                        normalization=normalization,
                                        block_type=block_type,
                                        upsampling=upsampling,
                                        skip_mode=skip_mode,
                                        dropout=dropout,
                                    )
        
        #instanciate the encoding conv_layers (conv_blocks + downscaling_layers)
        self.attention_blocks = nn.ModuleList()

        
        VisionMultiheadAttention(embed_size, decoder_dim, num_skip_channels, num_decoder_channels, num_heads=1)

        #construction loop initialization
        prev_shape = self.encoder_shapes[0]
        c_in = prev_shape[1]

        for c_out, skip_shape in zip(self.num_channels_list, self.encoder_shapes[1:]):
            pass

            
        

    def forward(self, x, skips):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size
        skips (list[torch.Tensor]): all the skip connections from the encoder

        Returns:
        x (torch.Tensor): output
        '''

        #reverse the skips
        skips = skips[::-1]

        #iterate over the number of blocks
        for i in range(self.num_blocks):

            #upscaling
            x = self.upscaling_layers[i](x)

            #get skip connection
            skip = skips[i]

            #add or append mode
            if self.skip_mode == 'append':
                x = torch.cat([skip, x], dim=1)
            elif self.skip_mode == 'add':
                x = x + skip
            else:
                raise NotImplementedError(f"{self.skip_mode} has not been implemented")

            #go through convolutional block
            x = self.conv_blocks[i](x)

        return x