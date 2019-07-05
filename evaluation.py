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

### IMPORT RESULTS
res_dir = "/home/sebwild/Desktop/promise_results"
basic_image_size = list(np.array(params.SINGLE_IMAGE_SIZE)//params.SUBSAMPLING)

### DEFINE DATASETS AND DATA LOADERS 
dataset_generator = dataset.MR_Dataset_Generator(n_splits = 5, i_split = 0)
test_dataset = dataset_generator.getTestDataset(0, 1)
test_size = test_dataset.__len__()

test_results = np.zeros(tuple([test_size] + basic_image_size), dtype=int)
for i in range(test_size):
    test_results[i,:,:,:] = np.load(os.path.join(res_dir, "test_" + str(i) + ".npy"))


images = np.zeros(tuple([test_size] + basic_image_size), dtype=int)
GT_labels = np.zeros(tuple([test_size] + basic_image_size), dtype=int)
for i in range(test_size):
    images[i,:,:,:] = test_dataset.__getitem__(i)[0].numpy()[0]
    GT_labels[i,:,:,:] = test_dataset.__getitem__(i)[1].numpy()[0]
    
dice_scores = np.array([losses.dice_score(test_results[i], GT_labels[i]) for i in range(test_size)])   

img_id = 9
slice_id_list = [7,10,13]

print("AVERAGE DICE:", np.mean(dice_scores))
print()
print("Original image index: ", test_dataset.indices[img_id])
print("dice = ", dice_scores[img_id])
comb = []
for slice_id in slice_id_list:
    margin_size = 5
    
    color_scale_GT = [0.3, 0.0, 0.0]
    image_GT = np.zeros((basic_image_size[1], basic_image_size[2], 3))
    for i in range(3):
        image_GT[:,:,i] = images[img_id,slice_id,:,:].astype(float)/np.max(images[img_id,slice_id,:,:])
        image_GT[:,:,i] = image_GT[:,:,i] + GT_labels[img_id,slice_id,:,:]*color_scale_GT[i]
    image_GT = image_GT/np.max(image_GT)
    
    color_scale_result = [0.3, 0.0 ,0.0]
    image_result = np.zeros((basic_image_size[1], basic_image_size[2], 3))
    for i in range(3):
        image_result[:,:,i] = images[img_id,slice_id,:,:].astype(float)/np.max(images[img_id,slice_id,:,:])
        image_result[:,:,i] = image_result[:,:,i] + test_results[img_id,slice_id,:,:]*color_scale_result[i]
    
    image_result = image_result/np.max(image_result)
    
    comb.append(np.concatenate((image_GT, np.ones((basic_image_size[1], margin_size, 3)), image_result), axis=1))
    comb.append(np.ones((margin_size, comb[-1].shape[1], 3)))

comb = np.concatenate(tuple(comb), axis=0)
plt.figure(dpi=200)
plt.axis('off')
plt.title("GT                       Seg", fontsize=8)
plt.imshow(comb) 
plt.savefig("/home/sebwild/Desktop/img_" + str(test_dataset.indices[img_id]) + ".jpg")
