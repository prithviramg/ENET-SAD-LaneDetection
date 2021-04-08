# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import tensorflow as tf
import numpy as np
import random
from PIL import Image

class Generator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, filePath, batch_size):
        'Initialization'
        self.filePath = filePath
        assert ("txt" in filePath), "please specify text file properly"
        self.dataFile = open(filePath,"r")
        self.lines = self.dataFile.readlines()
        self.batch_size = int(batch_size)
        self.on_epoch_end()
    def __len__(self):
        """ Size of the dataset.
        """
        return len(self.lines)//self.batch_size
    def on_epoch_end(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # determine the order of the images
        order = list(range(len(self.lines)))
        random.shuffle(order)
        # divide into groups, one group = one batch
        self.groups = [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
        
    def load_image_group(self,group):
        
        flag = True
        for eachSample in group:
            imagePath = self.lines[eachSample].split(",")[0]
            img = np.asarray(Image.open(imagePath).convert('RGB'))
            img = img.astype(np.float32)
            img /= 127.5
            img -= 1
            img = np.expand_dims(img,axis = 0)
            if flag == True:
                flag = False
                imageGroup = img
            else:
                imageGroup = np.concatenate((imageGroup,img),axis = 0)
            
        return imageGroup
            
        
    def load_target_group(self,group):
        
        flag = True
        for eachSample in group:
            targetPath = self.lines[eachSample].split(",")[1]
            targetPath = targetPath.replace("\n",'')
            img = np.asarray(Image.open(targetPath).convert('RGB'))
            img = img[:,:,:1]
            img = img.astype(np.float32)
            img /= 127.5
            img -= 1
            img[ img > 0] = 1
            img[ img <= 0] = 0
            img = np.expand_dims(img,axis = 0)
            if flag == True:
                flag = False
                targetGroup = img
            else:
                targetGroup = np.concatenate((targetGroup,img),axis = 0)
            
        return targetGroup
        
    def compute_input_output(self, group):
        """ Compute inputs and target outputs for the network.
        """
        # load images and annotations
        image_group       = self.load_image_group(group)
        target_group = self.load_target_group(group)


        return image_group, target_group

    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """
        group = self.groups[index]
        inputs, targets = self.compute_input_output(group)

        return inputs, targets