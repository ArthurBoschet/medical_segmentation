import torch.nn as nn

from segmentation import SegmentationModel
from encoders.conv_encoder import ConvEncoder
from decoders.conv_halfUnet_decoder import ConvHalfDecoder
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample


class HalfUNet(SegmentationModel):
    def __init__(
            self,
            input_shape,
            num_classes,
            num_channels_list,
            kernel_size=3,
            scale_factor=2,
            activation=nn.ReLU,
            normalization=nn.BatchNorm3d,
            block_type=DoubleConvBlock,
            downsampling=MaxPool3dDownscale,
            upsampling=TransposeConv3dUpsample,
            skip_mode='append',
            dropout=0,
            ):
        '''
        Implementation of a UNet model
        Parameters:
            input_shape (tuple): (C,D,H,W) of the input
            num_classes (int): number of classes in the segmentation
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            scale_factor (int): factor by which to downscale the image along depth, height and width and then rescale in decoder
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            downsampling (blocks.conv.downsampling.Downscale): downsampling scheme
            upsampling (blocks.conv.downsampling.Downscale): upsampling scheme
            skip_mode (str): one of 'append' | 'add' refers to how the skip connection is added back to the decoder path
            dropout (float): dropout added to the layer
            patch_size (int) : patch size for patch embedding
            channel_embedding (int) : number of channel for patch embedding
        '''

        super(HalfUNet, self).__init__()

        # encoder

        #encoder
        self.encoder = ConvEncoder(
                                input_shape,
                                num_channels_list,
                                kernel_size=kernel_size,
                                downscale_factor=scale_factor,
                                activation=activation,
                                normalization=normalization,
                                block_type=block_type,
                                downsampling=downsampling,
                                downscale_last=False,
                                dropout=dropout,
                            )

        # decoder
        self.decoder = ConvHalfDecoder(
            self.encoder.compute_output_dimensions(),
            num_channels_list[-2::-1],
            kernel_size=kernel_size,
            activation=activation,
            normalization=normalization,
            block_type=block_type,
            upsampling=upsampling,
            skip_mode=skip_mode,
            dropout=dropout,
        )

        # ouput layer (channelwise mlp) to have the desired number of classes
        self.output_layer = nn.Conv3d(
            num_channels_list[0],
            num_classes,
            1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections)
        x = self.output_layer(x)
        return x


import torch
# init default parameters
input = torch.rand((2, 1,32, 64, 64))
input_shape = input.shape[1:]
num_classes = 2
num_channels_list = [32, 64, 128]
kernel_size = 3
scale_factor = 2
activation = nn.LeakyReLU
normalization = nn.InstanceNorm3d
block_type = DoubleConvBlock
downsampling = MaxPool3dDownscale
upsampling = TransposeConv3dUpsample
skip_mode = "add"
dropout = 0.1

#torch device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"The device is {device}")

# init model

unet_model = HalfUNet(input_shape,
                  num_classes,
                  num_channels_list,
                  kernel_size=kernel_size,
                  scale_factor=scale_factor,
                  activation=activation,
                  normalization=normalization,
                  block_type=block_type,
                  downsampling=downsampling,
                  upsampling=upsampling,
                  skip_mode=skip_mode,
                  dropout=dropout)

'''
unet_model = UNet(input_shape,
                  num_classes,
                  num_channels_list,
                  kernel_size=kernel_size,
                  scale_factor=scale_factor,
                  activation=activation,
                  normalization=normalization,
                  block_type=block_type,
                  downsampling=downsampling,
                  upsampling=upsampling,
                  skip_mode=skip_mode,
                  dropout=dropout)
'''

output = unet_model(input.to(device))
print('input shape:', input.shape)
print('output shape unet:', output.shape)