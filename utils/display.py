import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid
from matplotlib.animation import FuncAnimation, PillowWriter

def unorm(x):
    # assume x (h,w,3)
    # Normalize an image to the range [0, 1].
    xmax = x.max((0,1))
    xmin = x.min((0,1))
    return(x - xmin)/(xmax - xmin)

def norm_all(store, n_t, n_s):
    #  Normalize all images across all timesteps and samples.
    nstore = np.zeros_like(store)
    for t in range(n_t):
        for s in range(n_s):
            nstore[t,s] = unorm(store[t,s])
    return nstore

def plot_sample(x_gen_store, n_sample, nrows, save_dir, fn,  w, save=False):
    '''
    x_gen_store: Array of generated images.
    n_sample: Number of samples.
    nrows: Number of rows in the plot grid.
    save_dir: Directory to save the animation.
    fn: Filename for the saved animation.
    w: Additional parameter for filename customization.
    save: Boolean to determine whether to save the animation.
    '''
    ncols = n_sample//nrows
    sx_gen_store = np.moveaxis(x_gen_store,2,4)                               # change to Numpy image format (h,w,channels) vs (channels,h,w)
    nsx_gen_store = norm_all(sx_gen_store, sx_gen_store.shape[0], n_sample)   # unity norm to put in range [0,1] for np.imshow
    
    # create gif of images evolving over time, based on x_gen_store
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=True, sharey=True,figsize=(ncols,nrows))
    def animate_diff(i, store):
        print(f'gif animating frame {i} of {store.shape[0]}', end='\r')
        plots = []
        for row in range(nrows):
            for col in range(ncols):
                axs[row, col].clear()
                axs[row, col].set_xticks([])
                axs[row, col].set_yticks([])
                plots.append(axs[row, col].imshow(store[i,(row*ncols)+col]))
        return plots
    ani = FuncAnimation(fig, animate_diff, fargs=[nsx_gen_store],  interval=200, blit=False, repeat=True, frames=nsx_gen_store.shape[0]) 
    plt.close()
    if save:
        savePath = f"{fn}_w{w}.gif" if w is not None else f"{fn}.gif"
        savePath = os.path.join(save_dir, savePath)
        ani.save(savePath, dpi=100, writer=PillowWriter(fps=5))
        print('saved gif at ' + savePath)
    return ani
