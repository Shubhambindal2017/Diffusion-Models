import sys
import torch
import Image
from torch import nn
import numpy as np

def prepareDataset(datasetName, datasetParams):
    if datasetName == 'sprites':
        dataset = SpritesDataset(datasetParams)
    elif datasetName == 'pokemon':
        dataset = PokemonDataset(datasetParams)
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

class PokemonDataset(torch.nn.utils.Dataset):
    def __init__(self, datasetParams):
        imgDir, labels, transform, null_context =  datasetParams['imgDir'], datasetParams['labels'], \
                                                                    datasetParams['transform'], datasetParams['null_context']
        
        self.imgDir = imgDir
        self.images = [img for img in os.listdir(imgDir) if img[0]!='.']
        print(f'images num : {len(self.images)}')

        if not null_context:
            self.labels = labels
            print(f'labels num : {self.labels.shape}')

        self.transform = transform
        self.null_context = null_context

    def __len__(self):
        return len(self.images)

    def __get_item__(self, idx):
        image = os.path.join(self.imgDir, self.images[idx])
        image = Image.open(image)
        if self.transform:
            image = self.transform(image)
        if self.null_context:
            label = torch.tensor(0).to(torch.int64)
        else:
            label = self.labels[index]
            label = torch.tensor(label).to(torch.int64)
        return (image, label)