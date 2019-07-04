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