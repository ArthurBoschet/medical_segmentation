import torch
import torch.nn as nn
import numpy as np

# personal modules
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample
from utils.conv_utils import conv3d_output_dim


class ConvHalfDecoder(nn.Module):
    def __init__(self,
                 encoder_shapes,
                 num_channels_list,
                 kernel_size=3,
                 activation=nn.ReLU,
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 upsampling=TransposeConv3dUpsample,
                 skip_mode='add',
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
        super(ConvHalfDecoder, self).__init__()

        assert kernel_size % 2 == 1, "kernel size should be an odd number (standard practice)"

        # set number of channels per blocks as well as number of blocks
        self.num_channels_list = num_channels_list
        self.num_blocks = len(num_channels_list)

        # reverse the order of encoder shapes
        self.encoder_shapes = encoder_shapes[::-1]
        assert len(
            self.encoder_shapes) == self.num_blocks + 1, "the number of blocks plus 1 must be the same as length of the encoder shapes (one encoder shape per block)"

        # skip connections parameters
        self.skip_mode = skip_mode
        # if self.skip_mode == 'add':
        #     for shape, c in zip(self.encoder_shapes[1:], self.num_channels_list):
        #         assert shape[
        #                    1] == c, f"the number of channels entered was {c} but {shape[1]} was expected based on encoder shapes in 'add' mode"

        # conv parameters for same convolution
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # instanciate the encoding upscaling_layers (conv_blocks + downscaling_layers)
        self.upscaling_layers = nn.ModuleList()

        # construction loop initialization
        prev_shape = self.encoder_shapes[0] # output shape
        c_in = prev_shape[1]

        final_shape = self.encoder_shapes[-1]
        final_c = self.num_channels_list[-1]
        for c_out, skip_shape in zip(self.num_channels_list, self.encoder_shapes[1:]):

            # we first do upscaling
            upscale_block = upsampling(prev_shape, final_shape, c_in, final_c)

            # update values
            c_in = c_out
            prev_shape = skip_shape

            # upsampling added
            self.upscaling_layers.append(upscale_block)


    def forward(self, x, skips):
        '''
        Parameters:
        x (torch.Tensor): (N,C_in,D,H,W) input size
        skips (list[torch.Tensor]): all the skip connections from the encoder

        Returns:
        x (torch.Tensor): output
        '''


        x = self.upscaling_layers[0](x)

        # reverse the skips
        skips = skips[::-1]

        # iterate over the number of blocks
        for i, skip in enumerate(skips[:]):

            # upscaling
            if i != len(skips)-1:
                skip = self.upscaling_layers[i+1](skip)


            # add or append mode
            if self.skip_mode == 'append':
                x = torch.cat([skip, x], dim=1)
            elif self.skip_mode == 'add':
                x = x + skip
            else:
                raise NotImplementedError(f"{self.skip_mode} has not been implemented")


        return x

    def compute_output_dimensions(self):
        '''
        computes the dimensions at the end of each convolutional block
        Returns:
            dimensions (List[Tuple]): dimension at the end of each convolutional block (first ones are skip connections while the last one is output of encoder)
        '''
        dimensions = []
        for channel, shape in zip(self.num_channels_list, self.encoder_shapes[1:]):
            dim = list(shape)
            dim[1] = channel
            dimensions.append(tuple(dim))

        return dimensions