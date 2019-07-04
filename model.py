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


def conv(n_in, n_out):
    return nn.Conv3d(n_in, n_out, (3,3,3), padding = (1,1,1))

def BN(n):
    return nn.BatchNorm3d(n)

def pool():
    return nn.MaxPool3d((2,2,2))

### DEFINING THE NET
class Net(nn.Module):
    def __init__(self, n_channels_FirstLayer = 8):
        super(Net, self).__init__()
        
        self.n = n_channels_FirstLayer  # number of feature channels in the first conv layer
        
        
        self.conv1_ = nn.Conv3d(1, self.n, (3,3,3), padding = (1,1,1))
        self.conv1 = nn.Conv3d(self.n, self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm1_ = nn.BatchNorm3d(self.n)
        self.batchnorm1 = nn.BatchNorm3d(self.n)
        self.pool1 = nn.MaxPool3d((2,2,2))
        self.conv2_ = nn.Conv3d(self.n, 2*self.n, (3,3,3), padding = (1,1,1))
        self.conv2 = nn.Conv3d(2*self.n, 2*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm2_ = nn.BatchNorm3d(2*self.n)
        self.batchnorm2 = nn.BatchNorm3d(2*self.n)
        self.pool2 = nn.MaxPool3d((2,2,2))
        self.conv3_ = nn.Conv3d(2*self.n, 4*self.n, (3,3,3), padding = (1,1,1))
        self.conv3 = nn.Conv3d(4*self.n, 4*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm3_ = nn.BatchNorm3d(4*self.n)
        self.batchnorm3 = nn.BatchNorm3d(4*self.n)
        self.pool3 = nn.MaxPool3d((2,2,2))
        self.conv4_ = nn.Conv3d(4*self.n, 8*self.n, (3,3,3), padding = (1,1,1))
        self.conv4 = nn.Conv3d(8*self.n, 8*self.n, (3,3,3), padding = (1,1,1))
        self.batchnorm4_ = nn.BatchNorm3d(8*self.n)
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
        
        layer1c_ = F.relu(self.batchnorm1_(self.conv1_(x)))
        layer1c = F.relu(self.batchnorm1(self.conv1(layer1c_)))
        layer1 = self.pool1(layer1c)
        layer2c_ = F.relu(self.batchnorm2_(self.conv2_(layer1)))
        layer2c = F.relu(self.batchnorm2(self.conv2(layer2c_)))
        layer2 = self.pool2(layer2c)
        layer3c_ = F.relu(self.batchnorm3_(self.conv3_(layer2)))
        layer3c = F.relu(self.batchnorm3(self.conv3(layer3c_)))
        layer3 = self.pool3(layer3c)
        layer4c_ = F.relu(self.batchnorm4_(self.conv4_(layer3)))
        layer4c = F.relu(self.batchnorm4(self.conv4(layer4c_)))
        
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