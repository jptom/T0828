# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 23:34:30 2020

@author: elevator-talk
"""
import random
import numpy as np
from tensorflow import keras
import pandas as pd
from yolo_v1 import read, C, B, GRID

class Generator(keras.utils.Sequence):
    def __init__(self, images:pd.DataFrame, labels:pd.DataFrame, batch_size, size=False):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.size=size
        
    def __len__(self):
        return (np.ceil(len(self.images)/float(self.batch_size))).astype(np.int)
    
    def __getitem__(self, idx):
        if idx > len(self):
            raise IndexError("Generator index out of range")
        l = self.batch_size if idx < len(self)-1 else len(self.images) - idx*self.batch_size 
        ret_image = []
        ret_label = []
        
        
        for i in range(l):
            
            image, label = read(self.images.iloc[idx*self.batch_size + i],
                                self.labels.iloc[idx*self.batch_size + i])
            ret_image.append(image)
            ret_label.append(label)
        if self.size:
            return np.array(ret_image), np.array(ret_label), self.images[idx*self.batch_size:(idx+1)*self.batch_size][['width', 'height']]
        else:    
            return np.array(ret_image), np.array(ret_label)
    

if __name__ == "__main__":
    from dataset import SVHN
    
    dataset = SVHN("dataset")
    x, y = dataset.train()
    generator = Generator(x, y, 64)
    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
    