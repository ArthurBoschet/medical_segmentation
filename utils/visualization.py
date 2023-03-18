import matplotlib.pyplot as plt
from ipywidgets import interact

def visualize_positional_encodings(tensor):

    @interact
    def visualise_4d_image(d=(0,tensor.shape[1]-1), h=(0,tensor.shape[2]-1), w=(0,tensor.shape[3]-1)): 
        #get image
        img = tensor[:, d, h, w].reshape(1, -1)

        #plot slice
        plt.figure(figsize=(15,10))
        plt.imshow(img, aspect=tensor.shape[0]/6)
        plt.xlabel('channels')
        plt.clim(-1,1)

        #show plot
        plt.show()