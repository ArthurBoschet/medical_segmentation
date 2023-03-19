import matplotlib.pyplot as plt
from ipywidgets import interact

def visualize_positional_encodings(pos_encoding, figsize=(15,10), title="Positional Encodings"):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(d=(0,pos_encoding.shape[1]-1), h=(0,pos_encoding.shape[2]-1), w=(0,pos_encoding.shape[3]-1)): 
        #get image
        img = pos_encoding[:, d, h, w].reshape(1, -1)

        #plot slice
        plt.figure(figsize=figsize)
        plt.imshow(img, aspect=pos_encoding.shape[0]/6)
        plt.xlabel('channels')
        plt.clim(-1,1)

        #show plot
        plt.show()



def visualize_attention(attention_weights_avg, figsize=(15,15), fraction=0.02, pad=0.05, title='Average Attention Weights'):
    '''
    shows a visualization of a positional encoding in a tensor
    Parameters:
        pos_encoding (torch.tensor): positional encoding tensor (C, D, H, W)
        figsize (Tuple[int]): size of the figure
        fraction (float): colorbar parameter
        pad (float): colorbar padding from the figure
        title (str): title of the plot
    '''

    @interact
    def visualise_4d_image(
        image=(0,attention_weights_avg.shape[0]-1),
        depth_decoder=(0,attention_weights_avg.shape[-3]-1),
        height_decoder=(0,attention_weights_avg.shape[-2]-1),
        width_decoder=(0,attention_weights_avg.shape[-1]-1),
        depth_skip=(0,attention_weights_avg.shape[1]-1),
        ): 
        #get image
        img = attention_weights_avg[image, depth_skip, :, :, depth_decoder, height_decoder, width_decoder]

        #plot slice
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(img)
        fig.colorbar(im, fraction=fraction, pad=pad)
        plt.title(title)

        #show plot
        plt.show()

def visualize_dataloaders(dataloader, resize=None, figsize=(10,5)):
    '''
    shows a visualization of a slice of an image and its label from a dataloader
    
    Args:
        dataloader: torch.utils.data.DataLoader
            dataloader to visualize
        image: int
            index of the image to visualize
        slice: int
            index of the slice to visualize
        figsize: tuple
            size of the figure
    '''
    @interact
    def plot_slice(image=(1, len(dataloader.dataset)), 
                   slice=(1, dataloader.dataset[0][0][0].numpy().shape[0])):
        # get image and label
        im = dataloader.dataset[image][0][0].numpy()
        label = dataloader.dataset[image][1][0].numpy()
        if slice > im.shape[0]-1:
            slice = im.shape[0]-1
        print("image shape:", im.shape)

        # plot slice
        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(im[slice], cmap='gray')
        ax[0].set_title('image')
        ax[1].imshow(label[slice], cmap='gray')
        ax[1].set_title('label')

        #show plot
        plt.show()
