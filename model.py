#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:51:32 2019

@author: sebwild
"""

import sys
import os
import numpy as np
import random
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import torch.optim as optim
from torchsummary import summary
import glob


# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)
if torch.cuda.is_available():
	os.environ["CUDA_VISIBLE_DEVICES"] = "1"
else:
	os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True

### DEFINING THE NET
class Net(nn.Module):
    def __init__(self, n_channels_FirstLayer = 8):
        super(Net, self).__init__()
        
        self.n = n_channels_FirstLayer  # number of feature channels in the first conv layer
        
        self.conv1 = nn.Conv3d(1, self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm1 = nn.BatchNorm3d(self.n)
        self.pool1 = nn.MaxPool3d((2,2,2))
        self.conv2 = nn.Conv3d(self.n, 2*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm2 = nn.BatchNorm3d(2*self.n)
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.conv3 = nn.Conv3d(2*self.n, 4*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm3 = nn.BatchNorm3d(4*self.n)
        self.pool3 = nn.MaxPool3d((2,2,2))
        self.conv4 = nn.Conv3d(4*self.n, 8*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm4 = nn.BatchNorm3d(8*self.n)
        
        self.conv4_up = nn.ConvTranspose3d(8*self.n, 4*self.n, (2,2,2), stride = (2,2,2))
        self.conv4d = nn.Conv3d(8*self.n, 4*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm4d = nn.BatchNorm3d(4*self.n)
        self.conv5_up = nn.ConvTranspose3d(4*self.n, 2*self.n, (2,2,2), stride = (2,2,2))
        self.conv5d = nn.Conv3d(4*self.n, 2*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm5d = nn.BatchNorm3d(2*self.n)
        self.conv6_up = nn.ConvTranspose3d(2*self.n, self.n, (2,2,2), stride = (2,2,2))
        self.conv6d = nn.Conv3d(2*self.n, self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm6d = nn.BatchNorm3d(self.n)
        
        self.FinalConv = nn.Conv3d(self.n, 2, (1,1,1))
    

    def forward(self, x):
        layer1c = F.relu(self.batchnorm1(self.conv1(x)))
        layer1 = self.pool1(layer1c)
        layer2c = F.relu(self.batchnorm2(self.conv2(layer1)))
        layer2 = self.pool2(layer2c)
        layer3c = F.relu(self.batchnorm3(self.conv3(layer2)))
        layer3 = self.pool3(layer3c)
        layer4c = F.relu(self.batchnorm4(self.conv4(layer3)))
        
        layer4x = self.conv4_up(layer4c)
        layer4d = torch.cat((layer3c, layer4x), 1)
        layer3x = F.relu(self.batchnorm4d(self.conv4d(layer4d)))
        
        layer3y = self.conv5_up(layer3x)
        layer3d = torch.cat((layer2c, layer3y), 1)
        layer2x = F.relu(self.batchnorm5d(self.conv5d(layer3d)))
        
        layer2y = self.conv6_up(layer2x)
        layer2d = torch.cat((layer1c, layer2y), 1)
        layer1x = F.relu(self.batchnorm6d(self.conv6d(layer2d)))

        final_layer = nn.Softmax(dim=1)(self.FinalConv(layer1x))

        return final_layer