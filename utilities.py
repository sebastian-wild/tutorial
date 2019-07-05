#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 22:40:31 2019

@author: sebwild
"""

import numpy as np


class AverageMeter(object):
    def __init__(self):
        self.elements = []
        self.avrg = 0.0
        
    def append(self, x):
        self.elements.append(x)
        self.avrg = np.mean(self.elements)
        
        
def get_learning_rate(optimizer):
	lr=[]
	for param_group in optimizer.param_groups:
		lr +=[ param_group['lr'] ]

	#assert(len(lr)==1) #we support only one param_group
	lr = lr[0]

	return lr