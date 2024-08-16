import os
import sys
import torch
from PIL import Image
from torch import nn
import numpy as np

def list_files_with_paths(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def prepareDataset(datasetName, datasetParams):
    if datasetName == 'sprites':
        dataset = SpritesDataset(datasetParams)
    elif datasetName == 'pokemon' or datasetName=='pokemon_large' or datasetName=='anime':
        dataset = PokemonDataset(datasetParams, datasetName)
    else:
       print()
       print(f'Dataset : {datasetName}, not yet supported')
       print() 
    return dataset

class SpritesDataset(torch.utils.data.Dataset):
    def __init__(self, datasetParams):
        imageFilename, labelFilename, transform, null_context = datasetParams['imageFilename'], datasetParams['labelFilename'], \
                                                                    datasetParams['transform'], datasetParams['null_context']
        self.sprites = np.load(imageFilename)
        self.labels = np.load(labelFilename)
        print(f'sprite num : {self.sprites.shape}')
        print(f'labels num : {self.labels.shape}')
        self.transform = transform
        self.null_context = null_context

    def __len__(self):
        return len(self.sprites)

    def __getitem__(self, index):
        image = self.sprites[index]
        if self.transform:
            image = self.transform(image)
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = self.labels[index]
            label = torch.tensor(label).to(torch.int64)
        return (image, label)

class PokemonDataset(torch.utils.data.Dataset):
    def __init__(self, datasetParams, datasetName):
        imgDir, labels, transform, null_context =  datasetParams['imgDir'], datasetParams['labels'], \
                                                                    datasetParams['transform'], datasetParams['null_context']
        

        self.imgDir = imgDir
        if datasetName=='pokemon':
            self.images = [os.path.join(self.imgDir, img) for img in os.listdir(imgDir) if img[0]!='.']
        elif datasetName=='pokemon_large' or datasetName=='anime':
            self.images = list_files_with_paths(self.imgDir)
            self.images = [img for img in self.images if '.svg' not in img]

        print(f'images num : {len(self.images)}')

        if not null_context:
            self.labels = labels
            print(f'labels num : {self.labels.shape}')

        self.transform = transform
        self.null_context = null_context

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = Image.open(image).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = self.labels[index]
            label = torch.tensor(label).to(torch.int64)
        return (image, label)