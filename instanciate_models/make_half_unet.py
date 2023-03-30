import torch.nn as nn
import torch
from models.blocks.conv_blocks import DoubleConvBlock
from models.blocks.downsampling import MaxPool3dDownscale
from models.blocks.upsampling import TransposeConv3dUpsample
from models.segmentation.halfunet import HalfUNet

def make_half_unet(dropout):
    '''
    Function that instanciates a half unet model
    Parameters:
        dropout (float): dropout rate
    Returns:
        unet_model (models.segmentation.unet.UNet): UNet model
    '''
    # init default parameters
    num_classes = 2
    input_shape = (1, 128, 128, 128)
    num_channels_list = [32, 64, 128, 256, 380, 512]
    kernel_size = 3
    scale_factor = 2
    activation = nn.LeakyReLU
    normalization = nn.InstanceNorm3d
    block_type = DoubleConvBlock
    downsampling = MaxPool3dDownscale
    upsampling = TransposeConv3dUpsample

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
                      dropout=dropout,
                      channel_ouputconv=64,
                      num_outputconv=2,
                      )
    
    return unet_model.to(device)