import torch
import torch.nn as nn
from positional_encodings.torch_encodings import PositionalEncodingPermute3D

class VisionMultiheadAttention(nn.Module):
  def __init__(self, embed_size, decoder_dim, num_skip_channels, num_decoder_channels, num_heads=1):
    '''
    Multiheaded attention block that can be used to enhance nn-Unet. 
    The query is the skip connection while the key and value are come 
    from the decoder path.

    Parameters:
      embed_size (int): Total dimension of the model
      num_skip_channels (int): number of channels in the skip connection
      num_decoder_channels (int): number of channels in the decoder path
      num_heads (int): number of attention heads
    '''
    super(VisionMultiheadAttention, self).__init__()

    #number of input channels
    self.num_skip_channels = num_skip_channels
    self.num_decoder_channels = num_decoder_channels

    #number of attention heads and embed size
    self.embed_size = embed_size
    self.decoder_dim = decoder_dim
    self.num_heads = num_heads
    
    #positional encoders
    self.query_pos_encoder = PositionalEncodingPermute3D(self.num_skip_channels)
    self.key_val_pos_encoder = PositionalEncodingPermute3D(self.num_decoder_channels)

    #multihead attention
    self.multihead_attention_block = nn.MultiheadAttention(self.embed_size, self.num_heads, batch_first=True, kdim=self.decoder_dim, vdim=self.decoder_dim)


  def forward(self, skip_path, decoder_path):
    '''
    Parameters:
      skip_path torch.Tensor: shape is (batch_size, num_skip_channels, skip_depth, skip_height, skip_width)
      decoder_path torch.Tensor: shape is (batch_size, num_decoder_channels, decoder_depth, decoder_height, decoder_width)
    '''
    #remember skip shape
    shape_skip = list(skip_path.shape)

    #add positional encodings to the skip and decoder paths 
    skip_path = skip_path + self.query_pos_encoder(skip_path)
    decoder_path = decoder_path + self.key_val_pos_encoder(decoder_path)

    #flatten the physical dimensions for both the skip path and decoder path
    skip_path = torch.flatten(skip_path, start_dim=2, end_dim=- 1)
    decoder_path = torch.flatten(decoder_path, start_dim=2, end_dim=- 1)

    #invert the length (height + width + depth) dimension with the channels
    skip_path = torch.flip(skip_path, [1,2])
    decoder_path = torch.flip(decoder_path, [1,2])

    #attention mechanism
    output, _ = self.multihead_attention_block(skip_path, decoder_path, decoder_path)

    #invert the length (height + width + depth) dimension with the channels
    output = torch.flip(output, [1,2])

    #reshape output
    output = torch.reshape(output, shape_skip)

    return output