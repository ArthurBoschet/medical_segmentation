import torch
from monai.networks.nets import SwinUNETR

def make_swin_unetr_baseline(dropout):
    '''
    Function that instanciates a swin_unetr baseline model
    Parameters:
        dropout (float): dropout rate
    Returns:
        swin_unet (models.segmentation.unet.UNet): UNet model
    '''
    # init default parameters
    num_classes = 2

    #torch device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"The device is {device}")

    # init model
    swin_unet = SwinUNETR(img_size=(128, 128, 128),
                          in_channels=1,
                          out_channels=num_classes,
                          feature_size=48,
                          drop_rate=dropout,
                          )
    
    return swin_unet.to(device)