import torch.nn as nn

#personal modules
from blocks.conv_blocks import SingleConvBlock, DoubleConvBlock, ResConvBlock
from blocks.downsampling import MaxPool3dDownscale, AvgPool3dDownscale
from utils.conv_utils import conv3d_output_dim

class ConvEncoder(nn.Module):
    def __init__(self,
                 input_shape,
                 num_channels_list,
                 kernel_size=3,
                 downscale_factor=2,
                 activation=nn.ReLU, 
                 normalization=nn.BatchNorm3d,
                 block_type=DoubleConvBlock,
                 downsampling=MaxPool3dDownscale,
                 dropout=0,
                 ):
        '''
        Convolutional encoder for UNet model. We assume that every convolution is a same convolution with no dilation.
        Parameters:
            input_shape (tuple): (C,D,H,W) of the input
            num_channels_list (list): list of number of channels in each block
            kernel_size (int or tuple): size of the convolving kernel must be an odd number
            downscale_factor (int): factor by which to downscale the image along depth, height and width
            activation (def None -> torch.nn.Module): non linear activation used by the block
            normalization (def int -> torch.nn.modules.batchnorm._NormBase): normalization
            block_type (blocks.conv_blocks.BaseConvBlock): one the conv blocks inheriting from the BaseConvBlock class
            downsampling (blocks.conv)
            dropout (float): dropout added to the layer
        '''
        super(ConvEncoder, self).__init__()

        assert kernel_size % 2 == 1, "kernel size should be an odd number (standard practice)"

        self.num_channels_list = num_channels_list
        self.num_blocks=len(num_channels_list)

        #make sure that shapes are compatible with architecture
        for i in range(1,4):
            assert input_shape[i] % downscale_factor**self.num_blocks == 0, "input shapes depth, height and width must be divisible by downscale_factor**num_blocks"

        #input shape (C, D, H, W)
        self.input_shape = input_shape

        #conv parameters
        self.kernel_size = kernel_size
        self.padding = kernel_size//2  

        #maxpooling parameters
        self.downscale_factor = downscale_factor

        #function that instanciates each block
        self.block_instanciator = lambda in_channels, out_channels: block_type(
                                                            in_channels, 
                                                            out_channels, 
                                                            kernel_size,  
                                                            padding=kernel_size//2,   
                                                            activation=activation, 
                                                            normalization=normalization,
                                                            dropout=dropout,
                                                        )
        
            
        #instanciate the encoding conv_layers (conv_blocks + downscaling_layers)
        self.conv_blocks = nn.ModuleList()
        self.downscaling_layers = nn.ModuleList()

        c_in = input_shape[0]
        for i, c_out in enumerate(self.num_channels_list):
            self.conv_blocks.append(self.block_instanciator(c_in, c_out))
            if i < self.num_blocks - 1:
                self.downscaling_layers.append(downsampling(self.downscale_factor))
        
        

    def forward(self, x):
        #we want to save the skip connections
        skip_connections = []

        #iterate over the number of blocks
        for i in range(self.num_blocks):

            #pass through convolutional block
            x = self.conv_blocks[i](x)

            #save skip connection
            skip_connections.append(x)

            #downscale unless last conv block
            if i < self.num_blocks - 1:
                x = self.downscaling_layers[i](x)

        return x, skip_connections
    
    def compute_output_dimensions(self):
        #output dimension at each 
        dimensions=[tuple([1] + list(self.input_shape))]

        for c_out in self.num_channels_list:
            dim = conv3d_output_dim(dimensions[-1], c_out, self.kernel_size, 1, self.padding, 1)
            dim = conv3d_output_dim(dim, c_out, self.downscale_factor, self.downscale_factor, 0, 1)
            dimensions.append()

        return dimensions


