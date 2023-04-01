#json
import json

#pytorch
import torch
import torch.nn as nn

#convolution blocks
from models.blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock, ResConvBlockUnetr, ConvNextBLock, DoubleConvNextBLock

#attention block
from models.blocks.attention_blocks import VisionMultiheadAttention

#downsampling
from models.blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale

#upsampling
from models.blocks.upsampling import InterpolateUpsample, TransposeConv3dUpsample

#models
from models.segmentation.unet import UNet
from models.segmentation.halfunet import HalfUNet
from models.segmentation.trans_unet import TransUNet
from models.segmentation.unet_convskip import UNetConvSkip
from monai.networks.nets import SwinUNETR

def make_model(config,  input_shape=(1, 128, 128, 128), num_classes=2):

    MODEL_DICT = {
        'UNet':UNet,
        'HalfUNet':HalfUNet,
        'TransUNet':TransUNet,
        'UNetConvSkip':UNetConvSkip,
        'SwinUNETR':SwinUNETR,
    }

    ACTIVATIONS = {
        'LeakyReLU':nn.LeakyReLU,
        'GELU':nn.GELU,
        'ReLU':nn.ReLU,
        'Identity':nn.Identity,
    }

    NORMALIZATION = {
        'InstanceNorm3d':nn.InstanceNorm3d,
        'BatchNorm3d':nn.BatchNorm3d,
    }

    CONV_BLOCKS = {
        'SingleConvBlock':SingleConvBlock,
        'DoubleConvBlock':DoubleConvBlock, 
        'ResConvBlock':ResConvBlock, 
        'ResConvBlockUnetr':ResConvBlockUnetr, 
        'ConvNextBLock':ConvNextBLock, 
        'DoubleConvNextBLock':DoubleConvNextBLock,
    }

    ATTENTION_BLOCKS = {
        'VisionMultiheadAttention':VisionMultiheadAttention,
    }

    DOWNSAMPLING_BLOCKS = {
        'MaxPool3dDownscale':MaxPool3dDownscale,
        'AvgPool3dDownscale':AvgPool3dDownscale,
    }

    UPSAMPLING_BLOCK = {
        'InterpolateUpsample':InterpolateUpsample,
        'TransposeConv3dUpsample':TransposeConv3dUpsample,
    }

    OPTIONS = {
        'append':'append',
        'add': 'add',
    }


    PARAMETERS_DICT = MODEL_DICT | ACTIVATIONS | NORMALIZATION | CONV_BLOCKS | ATTENTION_BLOCKS | DOWNSAMPLING_BLOCKS | UPSAMPLING_BLOCK | OPTIONS

    #get json config
    with open(config) as json_file:
        config = json.load(json_file)

    #device
    device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

    #parameters dict
    parameters = config["parameters"]
    parameters = {k:(PARAMETERS_DICT[v] if isinstance(v, str)  else v) for k,v in parameters.items()}

    #instanciate model
    try:
        model = MODEL_DICT[config['model']](**parameters, input_shape=input_shape, num_classes=num_classes)
    except:
        c,d,h,w = input_shape
        model = MODEL_DICT[config['model']](**parameters, img_size=(d,h,w), in_channels=c, out_channels=num_classes)

    return model.to(device)

