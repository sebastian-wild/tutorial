#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:53:56 2019

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
import model
import params

from scipy import ndimage

# set random seed
random.seed(2050)
np.random.seed(2050)
torch.manual_seed(2050)
torch.cuda.manual_seed_all(2050)

class augment(object):
    def __init__(self, img_input, label_input):
        self.img_input = img_input
        self.label_input = label_input
        
        print(self.img_input.shape)

class MR_Dataset(Dataset):
    def __init__(self, indices, normalize = "auto", normalize_mean = None, normalize_std = None, augment = False):
    
        self.indices = indices
        self.n_images = len(self.indices)
        
        self.normalize = normalize
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std
        self.single_image_size = params.SINGLE_IMAGE_SIZE
        self.augment = augment
        
        if params.HOST == "local":
            self.data_dir = r"/home/sebwild/Desktop/promise12_data/numpy_data"
        else:
            self.data_dir = r"/content/promise12"
    
        self.img = np.zeros([self.n_images, 1] + self.single_image_size, dtype = 'int16')
        self.label = np.zeros([self.n_images, 1] + self.single_image_size, dtype = 'int8')
        for i in range(self.n_images):
            self.img[i, 0, :, :, :] = np.load(os.path.join(self.data_dir, "image_train" + str(self.indices[i]).zfill(2) + ".npy"))
            self.label[i, 0, :, :, :] = np.load(os.path.join(self.data_dir, "label_train" + str(self.indices[i]).zfill(2) + ".npy"))
        self.img = self.img.astype("float32")
        self.label = self.label.astype("int")
                
            
        if self.normalize == "auto":
            assert self.normalize_mean is None
            assert self.normalize_std is None
            
            self.img_mean = np.mean(self.img)
            self.img_std = np.std(self.img)
            self.img = (self.img - self.img_mean)/self.img_std
            
        elif self.normalize == "explicit":
            assert not (self.normalize_mean is None)
            assert not (self.normalize_std is None)
            self.img = (self.img - self.normalize_mean)/self.normalize_std
        
        elif self.normalize is None:
            pass
    
        else:
            assert False
            
        self.img = self.img[:,:,::params.SUBSAMPLING, ::params.SUBSAMPLING, ::params.SUBSAMPLING] 
        self.label = self.label[:,:,::params.SUBSAMPLING, ::params.SUBSAMPLING, ::params.SUBSAMPLING]   
    
    def __len__(self):
        return self.n_images
    
    # outputs single image and label, each one of shape (1, 32, 128, 128) = (n_channels, depth, height, width)
    def __getitem__(self, i):  
        
        if not self.augment:
            return (torch.from_numpy(self.img[i, ...]), torch.from_numpy(self.label[i]))
        else:
            aug = augment(self.img[i, ...], self.label[i])
            return (self.img[i, ...], self.label[i])

class MR_Dataset_Generator(object):
    def __init__(self, n_splits = 5, i_split = 0):
        
        self.n_splits = n_splits
        self.i_split = i_split
            
        self.n_images_total = 50
        self.n_cv = self.n_images_total//self.n_splits

        self.perm = np.array([40, 13, 46, 47, 16, 34,  2, 20,  6, 15, 17, 33, 42,  7, 28, 27, 23,\
                                 9, 12, 11, 18, 30, 21, 22, 48,  0, 25, 39, 41, 14, 38, 24, 10, 29,\
                                 31, 49, 45,  3, 19, 37, 36, 26,  4, 35,  8, 32,  5, 44, 43,  1])
        
        
        self.test_indices = self.perm[(self.i_split*self.n_cv):((self.i_split+1)*self.n_cv)]
        self.train_indices = np.concatenate((self.perm[:(self.i_split*self.n_cv):] , \
                                                     self.perm[((self.i_split+1)*self.n_cv):]))
        
        assert np.all(np.sort(np.concatenate((self.train_indices, self.test_indices))) == np.arange(self.n_images_total))


    def getTrainDataset(self, normalize = "auto", augment = False):
        return MR_Dataset(self.train_indices, normalize = normalize, normalize_mean = None, normalize_std = None, augment = augment)
    
    def getTestDataset(self, train_img_mean, train_img_std):
        return MR_Dataset(self.test_indices, normalize = "explicit", normalize_mean = train_img_mean, normalize_std = train_img_std)
         
def plot_slice(dataset, image_id, slice_index = 15):
    plt.figure(dpi=100)    
    plt.axis('off')
    (img, label) = dataset.__getitem__(image_id)
    img = img.numpy()[0,slice_index,:,:]
    label = label.numpy()[0,slice_index,:,:]
    
    color_scale = [1.0, 0.0, 0.0]
    image_plot = np.zeros((img.shape[0], img.shape[1], 3))
    for i in range(3):
        image_plot[:,:,i] = img + label*color_scale[i]
    
    image_plot = image_plot - np.min(image_plot)
    image_plot = image_plot/np.max(image_plot)
    plt.imshow(image_plot)     




#dataset_generator = MR_Dataset_Generator(n_splits = 5, i_split = 0)
#train_dataset = dataset_generator.getTrainDataset()
#train_dataset_augment = dataset_generator.getTrainDataset(augment = True)
#
#img_id_tmp = 5
#slice_index_tmp = 8
#plot_slice(train_dataset, img_id_tmp, slice_index = slice_index_tmp)
#plot_slice(train_dataset_augment, img_id_tmp, slice_index = slice_index_tmp)

#img_orig = train_dataset.__getitem__(5)[0].numpy()
#
#offset = [0,0,-10]
#img_new = ndimage.interpolation.shift(img_orig, (0, int(offset[0]), int(offset[1]), int(offset[2])), mode='reflect')
#
#
#plt.figure(dpi=100)    
#plt.axis('off')
#plt.imshow(img_new[0, 15, :, :], cmap="gray")    