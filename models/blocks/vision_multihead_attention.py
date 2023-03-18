import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute3D

class VisionMultiheadAttention(nn.Module):
    def __init__(self, num_skip_channels, num_decoder_channels, embed_size=64, num_heads=8):
        '''
        Multiheaded attention block that can be used to enhance nn-Unet. 
        The query is the skip connection while the key and value are come 
        from the decoder path.

        Parameters:
            num_skip_channels (int): number of channels in the skip connection
            num_decoder_channels (int): number of channels in the decoder path
            embed_size (int): Total dimension of the model
            num_heads (int): number of attention heads
        '''
        super(VisionMultiheadAttention, self).__init__()

        #number of input channels
        self.num_skip_channels = num_skip_channels
        self.num_decoder_channels = num_decoder_channels

        #number of attention heads and embed size
        self.embed_size = embed_size
        self.num_heads = num_heads

        #space wise mlps
        self.linear1 = nn.Linear(self.num_skip_channels, self.embed_size)
        self.linear2 = nn.Linear(self.embed_size, self.num_skip_channels)

        #positional encoders
        self.query_pos_encoder = PositionalEncodingPermute3D(self.embed_size)
        self.key_val_pos_encoder = PositionalEncodingPermute3D(self.num_decoder_channels)

        #multihead attention
        self.multihead_attention_block = nn.MultiheadAttention(self.embed_size, self.num_heads, batch_first=True, kdim=self.num_decoder_channels, vdim=self.num_decoder_channels)


    def forward(self, skip_path, decoder_path, visualize=False):
        '''
        Parameters:
            skip_path (torch.Tensor): shape is (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
            decoder_path (torch.Tensor): shape is (batch_size, num_decoder_channels, decoder_depth, decoder_height, decoder_width)
            visualize (bool): whether or not to return attention weights in visualization format

        Returns:
            output (torch.Tensor): (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
        '''
        #remember skip shape (N, C_skip, D_skip, H_skip, W_skip)
        shape_skip = list(skip_path.shape)

        #remember skip shape (N, C_dec, D_dec, H_dec, W_dec)
        shape_decoder = list(decoder_path.shape)

        #expand channels
        skip_path = torch.transpose(skip_path, 1,4)
        skip_path = self.linear1(skip_path)
        skip_path = torch.transpose(skip_path, 1,4)

        #add positional encodings to the skip and decoder paths 
        skip_path = skip_path + self.query_pos_encoder(skip_path)
        decoder_path = decoder_path + self.key_val_pos_encoder(decoder_path)

        #flatten the physical dimensions for both the skip path and decoder path
        skip_path = torch.flatten(skip_path, start_dim=2, end_dim=- 1)
        decoder_path = torch.flatten(decoder_path, start_dim=2, end_dim=- 1)

        #invert the length (height + width + depth) dimension with the channels
        skip_path = torch.transpose(skip_path, 1,2)
        decoder_path = torch.transpose(decoder_path, 1,2)

        #attention mechanism
        output, attention_weights_avg = self.multihead_attention_block(skip_path, decoder_path, decoder_path)

        #contract channels of the output
        output = self.linear2(output)

        #invert the length (height + width + depth) dimension with the channels
        output = torch.transpose(output, 1,2)

        #reshape output and attention weights
        output = torch.reshape(output, shape_skip)

        #if visualize is set to true then reshape attention weights
        attention_weights_avg = self._reshape_attention(attention_weights_avg, shape_skip, shape_decoder)if visualize == True else None
        
        return output, attention_weights_avg


    def _reshape_attention(self, attention_weights_avg, shape_skip, shape_decoder):
        '''
        function that reshapes the attention from (N, L_skip, L_decoder) to (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        Parameters:
            attention_weights_avg (torch.tensor): attention weigths with shape (N, L_skip, L_decoder)
            shape_skip (list[int]): shape of the skip connection (query)
            shape_decoder (list[int]): shape of the decoder context (key and value)

        Returns:
            attention_weights_avg (torch.tensor): attention weigths with shape (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        '''
        #detach from the graph
        attention_weights_avg = attention_weights_avg.detach()

        # (N, L_skip, L_decoder) -> (N, L_decoder, L_skip)
        attention_weights_avg = torch.transpose(attention_weights_avg, -1, -2)
        attention_shape = list(attention_weights_avg.shape)

        #(N, L_decoder, L_skip) -> (N, L_decoder, D_skip, H_skip, W_skip)
        attention_weights_avg = torch.reshape(attention_weights_avg, attention_shape[:-1] + shape_skip[2:])

        #(N, L_decoder, D_skip, H_skip, W_skip) -> (N, D_skip, H_skip, W_skip, L_decoder)
        attention_weights_avg = torch.permute(attention_weights_avg, (0,2,3,4,1))
        attention_shape = list(attention_weights_avg.shape)

        #(N, D_skip, H_skip, W_skip, L_decoder) -> (N, D_skip, H_skip, W_skip, D_dec, H_dec, W_dec)
        attention_weights_avg = torch.reshape(attention_weights_avg, attention_shape[:-1] + shape_decoder[2:])

        return attention_weights_avg