import matplotlib.pyplot as plt
from ipywidgets import interact

def visualize_positional_encodings(pos_encoding):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
    '''

    @interact
    def visualise_4d_image(d=(0,pos_encoding.shape[1]-1), h=(0,pos_encoding.shape[2]-1), w=(0,pos_encoding.shape[3]-1)): 
        #get image
        img = pos_encoding[:, d, h, w].reshape(1, -1)

        #plot slice
        plt.figure(figsize=(15,10))
        plt.imshow(img, aspect=pos_encoding.shape[0]/6)
        plt.xlabel('channels')
        plt.clim(-1,1)

        #show plot
        plt.show()