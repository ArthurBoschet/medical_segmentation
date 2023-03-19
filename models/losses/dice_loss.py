import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        '''
        Implementation of the dice loss taken from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
        '''
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        '''
        Parameters:
            inputs (torch.tensor): input from the model
            targets (torch.tensor): target tensor
        '''
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice