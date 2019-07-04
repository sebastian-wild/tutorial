#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 16:52:01 2019

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
import model
import dataset
import params
from utilities import AverageMeter
import losses

''' 
TO DO:
    
    toGPU stuff in case we are on google
'''


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



    
print("Running on HOST = " + params.HOST)



### DEFINE DATASETS AND DATA LOADERS 
dataset_generator = dataset.MR_Dataset_Generator(n_splits = 5, i_split = 0)
train_dataset = dataset_generator.getTrainDataset()
test_dataset = dataset_generator.getTestDataset(train_dataset.img_mean, train_dataset.img_std)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params.BATCH_SIZE,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1,
                                          shuffle=False, num_workers=2)


### DEFINE NET ARCHITECTURE
net = model.Net(n_channels_FirstLayer = params.N_CHANNELS_FIRST_LAYER)
if torch.cuda.is_available():
    net.cuda()
if params.HOST == "local":
    summary(net, input_size=tuple([1] + list(np.array(params.SINGLE_IMAGE_SIZE)//params.SUBSAMPLING)))


### DEFINING THE LOSS FUNCTION AND OPTIMIZER
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
loss_fct = lambda outputs, labels : losses.dice_loss(outputs, labels[:,0,:,:,:], power = params.DICE_POWER)
#loss_fct = lambda outputs, labels : losses.cross_entropy_custom(outputs, labels[:,0,:,:,:])


### TRAIN THE NETWORK
for epoch in range(25):  # loop over the dataset multiple times
    net.train()
    train_loss = AverageMeter()
    for i, data in enumerate(train_loader, 0):
        
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        # forward + backward + optimize
        outputs = net(inputs)
        
        loss_train_tmp = loss_fct(outputs, labels)
        
    
        optimizer.zero_grad()
        loss_train_tmp.backward()
        optimizer.step()

        train_loss.append(loss_train_tmp.item())
        
        #print('[%d] train loss: %.3f' %
        #      (i + 1, train_loss.avrg))

    if torch.cuda.is_available():
        net.cuda()
        
    net.eval()
    test_loss = AverageMeter()
    with torch.no_grad():    
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs = net(inputs)
            loss_test_tmp = loss_fct(outputs, labels)
            test_loss.append(loss_test_tmp.item())
        #print("                      test loss: %.3f" % test_loss.avrg )
     
    print("epoch " + str(epoch+1) + ": %.3f, %.3f" % (train_loss.avrg, test_loss.avrg))        
        
print('Finished Training')