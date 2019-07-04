#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 21:05:21 2019

@author: sebwild
"""

import os
import sys

if os.path.isdir("/home/sebwild/Desktop/promise12_data"):
    HOST = "local"
else:
    HOST = "google"
    
print("HOST = " + HOST)

SINGLE_IMAGE_SIZE = [32, 128, 128] # D, H, W
SUBSAMPLING = 2
N_CHANNELS_FIRST_LAYER = 16
BATCH_SIZE = 4

DICE_POWER = 2