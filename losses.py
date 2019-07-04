#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:44:08 2019

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
#import model
#import params

# set random seed
#random.seed(2050)
#np.random.seed(2050)
#torch.manual_seed(2050)
#torch.cuda.manual_seed_all(2050)


def dice_loss(outputs, labels, power = 2, eps = 1e-7):
    label_float = labels.to(outputs.dtype)
    
    num = 2*torch.sum(outputs[:,1,:,:,:]*label_float, dim = [1,2,3])
    denom = torch.sum(torch.pow(outputs[:,1,:,:,:], power) + label_float, dim = [1,2,3])
    return 1 - torch.mean(num/(denom + eps))
    



def cross_entropy_custom(outputs, labels):
    labels_float = labels.to(outputs.dtype)
    # softmax
    #outputs_exp = torch.exp(outputs)
    #exp_sum = torch.sum(outputs_exp, dim = 1)[:,None,:,:,:]
    #softmax = outputs_exp/exp_sum

    return torch.mean(-labels_float*torch.log(outputs[:,1,:,:,:]) - (1 - labels_float)*torch.log(outputs[:,0,:,:,:]))
    
    
#outputs_test = torch.from_numpy(np.random.rand(4,2,16,32,32))    
#labels_test = torch.from_numpy(np.random.randint(2, size = (4,16,32,32)))
#
#loss_test = cross_entropy_custom(outputs_test, labels_test)
#loss_default = nn.CrossEntropyLoss()(outputs_test, labels_test)
#
#print(loss_test.item())
#print(loss_default.item())


#p_img_test = np.log(np.random.rand(10,2,4,4))
#label_test = np.random.randint(2, size = (10,2,4,4))
#print(cross_entropy_np(p_img_test, label_test))  
#
#p_img_torch = torch.from_numpy(p_img_test)
#label_torch = torch.from_numpy(label_test)
#print(cross_entropy(p_img_torch, label_torch).item())