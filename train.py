'''
Understand DDPM
'''
import torch
from torch import nn
from torch.utils.data import DataLoader


device = 'cuda' # 'cuda' or 'cpu'