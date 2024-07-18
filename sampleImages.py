import os
import sys
import tqdm
import torch
import numpy as np
from IPython.display import HTML
import matplotlib.pyplot as plt
from utils.UNetScratch import UNet
from utils.samplers import DDPMVanilla
from utils.display import plot_sample

device = 'cpu'
expName = 'starter_t_500'
expSubName = None # should be a string or None
model_architecture = 'UNetScratch'
model_path = '/teamspace/studios/this_studio/diffusion-models/output/training/train_sprite_epochs_40/models/model_32.pth'
samplerAlgo = 'DDPMVanilla'
timesteps = 500
h = 16 # height
n_sample = 32
saveIntermediate = True

if saveIntermediate:
    save_dir = 'output/generateSprites'
    fileName = expName
    subFileName = expSubName
    os.makedirs(save_dir, exist_ok=True)


def allowChangeInParam():
    #y_n = input('Check parameters!! Want to continue?? (Y/N)\n')
    y_n = 'Y'
    if y_n.lower().strip() in ['y', 'yes']:
        print(f'Thanks!! Continuing......')
    else:
        print(f'Thanks!! Update Parameters.....')
        sys.exit(0)

if model_architecture=='UNetScratch':
    in_channels = 3 
    c_size = 5
    f_size = 64
    h = h
    downsampleList              =  [ 2, 2, 4]
    inChannels_outChannelsList  =  [(f_size, f_size), (f_size, 2*f_size), (2*f_size, 2*f_size)]
    
    print(f'Parameters to be used : ')
    print(f'\t in_channels : {in_channels}')
    print(f'\t c_size : {c_size}')
    print(f'\t f_size : {f_size}')
    print(f'\t h : {h}')
    print(f'\t downsampleList : {downsampleList}')
    print(f'\t inChannels_outChannelsList : {inChannels_outChannelsList}')
    print()
    allowChangeInParam()
    model = UNet(in_channels, c_size, f_size, h, downsampleList, inChannels_outChannelsList).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device) )
    model.eval()

if samplerAlgo=='DDPMVanilla':
    sampler = DDPMVanilla(timesteps, beta1=1e-4, beta2=0.02, device=device)
    samples, intermediate = sampler.sample_ddpm(model, h, n_sample, save_rate=20)

# visualize samples
plt.clf()
animation = plot_sample(intermediate,32,4,save_dir, fileName, subFileName, save=saveIntermediate)
HTML(animation.to_jshtml())
