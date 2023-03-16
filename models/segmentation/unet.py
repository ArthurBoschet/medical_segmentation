import torch
import torch.nn as nn 

class UNet(nn.Module):
    def __init__(self, 
                 input_dim, 
                 scale_factor=2,
                 ):
        '''
        Implementation of a UNet model
        Parameters:
            input_dim (Tuple[int, int, int]): must refer to depth, height & width
        '''
        super(UNet, self).__init__()
        pass
        
        
