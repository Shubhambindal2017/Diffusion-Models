import sys
import torch
from torch import nn

def prepareDataset(datasetName):
    if datasetName == 'spirites':
        dataset = SpiritesDataset()
    else:
       print()
       print(f'Dataset : {datasetName}, not yet supported')
       print() 
    return dataset
