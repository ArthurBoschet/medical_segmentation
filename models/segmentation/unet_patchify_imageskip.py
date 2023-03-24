import torch.nn as nn

from segmentation.segmentation import SegmenationModel
from encoders.conv_swinpatch_encoder import ConvPatchEncoder
from decoders.conv_decoder import ConvDecoder
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample


class UNetPatch(SegmenationModel):
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
            patch_size=patch_size,
            channel_embedding=channel_embedding
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
        super(UNetPatch, self).__init__()


        # encoder
        self.encoder = ConvPatchEncoder(
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
            patch_size = patch_size,
            channel_embedding = channel_embedding
        )

        # decoder
        self.decoder = ConvDecoder(
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



