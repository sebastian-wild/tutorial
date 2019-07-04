#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 17:30:32 2019

@author: sebwild
"""

import os
import glob
import sys
import numpy as np



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
from matplotlib import pyplot as plt


# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

### IMPORT RESULTS
res_dir = "/home/sebwild/Desktop/promise_results"

test_results = np.zeros((10,16,64,64), dtype=int)
for i in range(10):
    test_results[i,:,:,:] = np.load(os.path.join(res_dir, "test_" + str(i) + ".npy"))

### DEFINE DATASETS AND DATA LOADERS 
dataset_generator = dataset.MR_Dataset_Generator(n_splits = 5, i_split = 0)
test_dataset = dataset_generator.getTestDataset(0, 1)

images = np.zeros((10,16,64,64), dtype=int)
GT_labels = np.zeros((10,16,64,64), dtype=int)
for i in range(10):
    images[i,:,:,:] = test_dataset.__getitem__(i)[0].numpy()[0]
    GT_labels[i,:,:,:] = test_dataset.__getitem__(i)[1].numpy()[0]
    
dice_scores = np.array([losses.dice_score(test_results[i], GT_labels[i]) for i in range(10)])   

img_id = 4
slice_id = 8

plt.figure()
plt.imshow(images[img_id,slice_id,:,:],cmap='gray') 
plt.figure()
plt.imshow(GT_labels[img_id,slice_id,:,:],cmap='gray') 
plt.figure()
plt.imshow(test_results[img_id,slice_id,:,:],cmap='gray') 
