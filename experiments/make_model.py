import sys
sys.path.append('/Users/arthurboschet/ift-6759-project/')
sys.path.append('/Users/arthurboschet/ift-6759-project/models/')

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
    'InstanceNorm3d':nn.BatchNorm3d,
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

#baselines
config_unet = {
    'model':'UNet',
    'training':{
        'lr':1e-3,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64], #, 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'DoubleConvBlock',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
        }
}

config_swin = {
    'model':'SwinUNETR',
    'training':{
        'lr':1e-3,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            "feature_size":48,
    }
}

# conv blocks
config_resblock1 = {
    'model':'UNet',
    'training':{
        'lr':1e-4,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64],# 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'ResConvBlockUnetr',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
        }
}

config_resblock2 = {
    'model':'UNet',
    'training':{
        'lr':1e-4,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64],# 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'ResConvBlock',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
        }
}

config_convnext = {
    'model':'UNet',
    'training':{
        'lr':1e-4,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64], # 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'ConvNextBLock',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
        }
}

#architectures
config_halfunet = {
    'model':'HalfUNet',
    'training':{
        'lr':1e-3,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64], # 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'DoubleConvBlock',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
        }
}

config_transunet = {
    'model':'TransUNet',
    'training':{
        'lr':1e-3,
        'weight_decay':1e-7,
        'factor':0.1,
        'patience':20,
        'batch_size':2,
        'resize':(128, 128, 128),
    },
    'parameters':{
            'num_channels_list':[32, 64], # 128, 256, 380, 512],
            'kernel_size':3,
            'scale_factor':2,
            'activation':'LeakyReLU',
            'normalization': 'InstanceNorm3d',
            'block_type':'DoubleConvBlock',
            'downsampling':'MaxPool3dDownscale',
            'upsampling':'TransposeConv3dUpsample',
            'skip_mode':'append',
            'patch_size_factor':8,
            'embed_size':64, 
            'num_heads':8,
            'activation_attention_embedding':'Identity',
            'normalization_attention':'BatchNorm3d',
            'upscale_attention':'TransposeConv3dUpsample',
            'dropout_attention':0,
        }
}

configs = [
    ('UNet', config_unet), 
    ('swin', config_swin), 
    ('resblock1', config_resblock1), 
    ('resblock2', config_resblock2), 
    ('convnext', config_convnext),
    ('transunet', config_transunet),
    ('halfunet', config_halfunet),
    
    ]


def make_model(config,  input_shape=(1, 128, 128, 128), num_classes=2):

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
    

print("################################## unet ##################################")
print(make_model(config_unet, input_shape=(1, 32, 32, 32)))
print("\n\n\n")

print("################################## swin ##################################")
print(make_model(config_swin, input_shape=(1, 32, 32, 32)))
print("\n\n\n")