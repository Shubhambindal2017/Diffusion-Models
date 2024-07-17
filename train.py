import os
import sys
import tqdm
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.UNetScratch import UNet
from utils.samplers import DDPMVanilla
from utils.display import plot_sample
from utils.datasets import prepareDataset

device = 'cpu'
expName = 'train_sprite_epochs_40'
datasetName = ''
model_architecture = 'UNetScratch'
samplerAlgo = 'DDPMVanilla'
timesteps = 500
h = 16 # height
batch_size = 128               # batch_size
learning_rate = 1e-3
saveIntermediate = True

if saveIntermediate:
    epoch_save_rate = 2
    save_models_at_epoch_save_rate = True
    n_sample = 32
    save_dir = f'output/training/{expName}'

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
    #model.load_state_dict(model_path, map_location=device)
    model.eval()

if samplerAlgo=='DDPMVanilla':
    sampler = DDPMVanilla(timesteps, beta1=1e-4, beta2=0.02, device=device)
    perturb_noise = sampler.perturb_noise

train()

def getDataLoader():
    dataset = prepareDataset(datasetName)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

def train():

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = getDataLoader()

    for epoch in range(1, epochs+1):
        model.train()

        print(f'Epoch : {epoch}')

        dataIterator = tqdm(dataloader, min_interval=2)

        for x, _ in dataIterator:
            optim.zero_grad()
            
            t = torch.randint(1, timesteps+1, (x.shape[0],)).to(device)
            x = x.to(device)
            noise = torch.randn_like(x)

            x_pertb = perturb_noise(x, t, noise)
            pred_noise = model(x_pertb, t/timesteps)

            loss = F.mse_loss(pred_noise, noise)
            loss.backward()
            optim.step()

        if epoch%epoch_save_rate==0 or epoch == epochs:
            if not os.path.exists(save_dir):
                os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
                os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)

            if save_models_at_epoch_save_rate or (not save_models_at_epoch_save_rate and epoch == epochs):
                ## save models at epoch_save_rate or save only last model
                torch.save(model.state_dict(), os.path.join(save_dir, 'models', f"model_{epoch}.pth"))
                print(f'Model saved at : {os.path.join(save_dir, "models", f"model_{epoch}.pth")}')

            mode.eval()
            samples, intermediate = sampler.sample_ddpm(model, h, n_sample, save_rate=20)
            animation = plot_sample(intermediate,32,4,os.path.join(save_dir, 'output'), fileName=f'epoch_{epoch}', subFileName=None, save=saveIntermediate)