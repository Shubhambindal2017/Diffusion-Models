import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision import models, transforms
from utils.UNetScratch import UNet
from utils.samplers import DDPMVanilla
from utils.display import plot_sample
from utils.datasets import prepareDataset
from torchsummary import summary

import logging
from torch.utils.tensorboard import SummaryWriter

device = 'cuda'
expName = 'anime_train_exp_3'
datasetName = 'anime' # sprites # pokemon_large
model_architecture = 'UNetScratch'
samplerAlgo = 'DDPMVanilla'
timesteps = 500
## SEE ARCHITECTURE PARAMS TOO - especially h and batch_size ; other Unet params too as per dataset
learning_rate = 1e-3
epochs = 64
saveIntermediate = True

if saveIntermediate:
    epoch_save_rate = 4
    save_models_at_epoch_save_rate = False #True
    n_sample = 32
    save_dir = f'output/training/{expName}'

def allowChangeInParam(flag):
    if flag=='input_params':
        y_n = input('Check parameters!! Want to continue?? (Y/N)\n')
    elif flag=='summary':
        y_n = input('Check model summary!! Want to continue?? (Y/N)\n')
    #y_n = 'Y'
    if y_n.lower().strip() in ['y', 'yes']:
        print(f'Thanks!! Continuing......')
    else:
        print(f'Thanks!! Update Parameters.....')
        sys.exit(0)

if model_architecture=='UNetScratch':
    if datasetName == 'sprites':
        h = 16 # height
        batch_size = 128               # batch_size`
        in_channels = 3 
        c_size = 5
        f_size = 64
        h = h
        downsampleList              =  [ 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), (f_size, 2*f_size), (2*f_size, 2*f_size)]

    if datasetName == 'pokemon' or datasetName == 'pokemon_large':
        h = 64 #16 # height
        batch_size = 32               # batch_size`
        in_channels = 3 
        c_size = 5
        f_size = 32
        h = h
        downsampleList              =  [ 2, 2, 4, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), # 32, 32 -> 32
                                        (f_size, 2*f_size), # 32, 64 -> 16
                                        (2*f_size, 4*f_size), # 64, 128 -> 4
                                        (4*f_size, 4*f_size),] # 128, 128 -> 1

        '''
        h = 32
        batch_size = 32             # batch_size`- wth L4 instance
        in_channels = 3 
        c_size = 5
        f_size = 32 # let's want to make hidden embd of 256
        h = h
        downsampleList              =  [ 2, 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), # 32, 32 -> 16
                                        (f_size, 2*f_size), # 32, 64 -> 8
                                        (2*f_size, 4*f_size), # 64, 128 -> 4
                                        (4*f_size, 4*f_size)] # 128, 128 -> 1
        '''
        '''
        downsampleList              =  [ 2, 2, 2, 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), # 32, 32 -> 64
                                        (f_size, 2*f_size), # 32, 64 -> 32
                                        (2*f_size, 4*f_size), # 64, 128 -> 16
                                        (4*f_size, 8*f_size), # 128, 256 -> 8
                                        (8*f_size, 8*f_size), # 256, 256 -> 4
                                        (8*f_size, 8*f_size),] # 256, 256 -> 1
        '''
    if datasetName == 'anime':
        h = 32 #16 # height
        batch_size = 128               # batch_size`
        in_channels = 3 
        c_size = 5
        f_size = 64
        h = h
        downsampleList              =  [ 2, 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, 2*f_size), # 64, 128 -> 16
                                        (2*f_size, 4*f_size), # 128, 256 -> 8
                                        (4*f_size, 8*f_size), # 256, 512 -> 4
                                        (8*f_size, 8*f_size),] # 512, 512 -> 1
        '''
        h = 32 #16 # height
        batch_size = 128               # batch_size`
        in_channels = 3 
        c_size = 5
        f_size = 64
        h = h
        downsampleList              =  [ 2, 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), # 64, 64 -> 16
                                        (f_size, 2*f_size), # 64, 128 -> 8
                                        (2*f_size, 4*f_size), # 128, 256 -> 4
                                        (4*f_size, 4*f_size),] # 256, 256 -> 1
        '''
        '''
        h = 32 #16 # height
        batch_size = 128               # batch_size`
        in_channels = 3 
        c_size = 5
        f_size = 32 #64
        h = h
        downsampleList              =  [ 2, 2, 2, 4]
        inChannels_outChannelsList  =  [(f_size, f_size), # 32, 32 -> 16
                                        (f_size, 2*f_size), # 32, 64 -> 8
                                        (2*f_size, 4*f_size), # 64, 128 -> 4
                                        (4*f_size, 4*f_size),] # 128, 128 -> 1
        '''
    

    print(f'Parameters to be used : ')
    print(f'\t in_channels : {in_channels}')
    print(f'\t c_size : {c_size}')
    print(f'\t f_size : {f_size}')
    print(f'\t h : {h}')
    print(f'\t downsampleList : {downsampleList}')
    print(f'\t inChannels_outChannelsList : {inChannels_outChannelsList}')
    print()
    allowChangeInParam('input_params')
    model = UNet(in_channels, c_size, f_size, h, downsampleList, inChannels_outChannelsList).to(device)
    summary(model, [(in_channels, h, h), (1, 1, 1), (c_size, 1, 1)])
    allowChangeInParam('summary')

if samplerAlgo=='DDPMVanilla':
    sampler = DDPMVanilla(timesteps, beta1=1e-4, beta2=0.02, device=device)
    perturb_noise = sampler.perturb_noise

if datasetName == 'sprites':
    transform = transforms.Compose([
        transforms.ToTensor(),                # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]

    ])
    datasetParams = {}
    datasetParams['imageFilename'] = '/teamspace/studios/this_studio/diffusion-models/data/sprites/sprites_1788_16x16.npy'
    datasetParams['labelFilename'] = '/teamspace/studios/this_studio/diffusion-models/data/sprites/sprite_labels_nc_1788_16x16.npy'
    datasetParams['transform'] = transform
    datasetParams['null_context'] = False

elif datasetName == 'pokemon':
    transform = transforms.Compose([
        transforms.Resize(size=(h, h)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(), # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
    datasetParams = {}
    datasetParams['imgDir'] = '/teamspace/studios/this_studio/diffusion-models/data/pokemon/pokemon'
    datasetParams['labels'] = None
    datasetParams['transform'] = transform
    datasetParams['null_context'] = True

elif datasetName == 'pokemon_large':
    '''
    transform = transforms.Compose([
        transforms.Resize(size=(h, h)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=5),
        transforms.ToTensor(), # from [0,255] to range [0.0,1.0]
        transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    ])
    '''
    transform = transforms.Compose([
        transforms.Resize(size=(64, 64)),  # Increased image size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-15, 15)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    datasetParams = {}
    datasetParams['imgDir'] = '/teamspace/studios/this_studio/diffusion-models/data/pokemon_large/PokemonData'
    datasetParams['labels'] = None
    datasetParams['transform'] = transform
    datasetParams['null_context'] = True

elif datasetName == 'anime':
    transform = transforms.Compose([
        transforms.Resize(size=(32, 32)),  # Increased image size
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    datasetParams = {}
    datasetParams['imgDir'] = '/teamspace/studios/this_studio/diffusion-models/data/animefacedataset'
    datasetParams['labels'] = None
    datasetParams['transform'] = transform
    datasetParams['null_context'] = True

def getDataLoader():
    dataset = prepareDataset(datasetName, datasetParams)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

def train():
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=os.path.join(save_dir, 'runs/experiment1'))

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    dataloader = getDataLoader()

    for epoch in range(1, epochs+1):
        model.train()

        print(f'Epoch : {epoch}')

        dataIterator = tqdm(dataloader, mininterval=2)

        loss_list = []
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
            loss_list.append(loss.item())

        # Log loss
        logging.info(f'Epoch: {epoch}, Loss: {round(np.mean(loss_list), 3)}')
        print(f'Epoch: {epoch}, Loss: {round(np.mean(loss_list), 3)}')
        # Log loss to TensorBoard
        writer.add_scalar('Loss/train', round(np.mean(loss_list), 3), epoch)


        if epoch==1 or epoch%epoch_save_rate==0 or epoch == epochs:
            os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
            os.makedirs(os.path.join(save_dir, 'output'), exist_ok=True)

            if save_models_at_epoch_save_rate or (not save_models_at_epoch_save_rate and epoch == epochs):
                ## save models at epoch_save_rate or save only last model
                torch.save(model.state_dict(), os.path.join(save_dir, 'models', f"model_{epoch}.pth"))
                print(f'Model saved at : {os.path.join(save_dir, "models", f"model_{epoch}.pth")}')

            model.eval()
            samples, intermediate = sampler.sample_ddpm(model, h, n_sample, save_rate=20)
            animation = plot_sample(intermediate,32,4,os.path.join(save_dir, 'output'), f'epoch_{epoch}', None, save=saveIntermediate)

    writer.close()

train()