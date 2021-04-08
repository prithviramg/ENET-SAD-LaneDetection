# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import sys
import os
import tensorflow as tf
sys.path.insert(1, os.path.dirname(os.getcwd()))
    
from models.encoder import encoder_build
from models.decoder import decoder_build
from utils.distillation import compute_distillation
from preprocessing.dataloader import Generator


########################## Data Input ###############################
trainDataPath = "D:\Downloads\Tusimple\data.txt"
valDataPath = "/workspace/work/Tusimple_dataset/train_set/val.txt"
batchSize = 8
epochs = 50
snapshotPath = "/workspace/work/enet_sad_naiveresize/snapshot"
tensorboardPath = "/workspace/work/enet_sad_naiveresize/tensorboard"

########################## Data Input ###############################


inputs = tf.keras.layers.Input(shape=(None, None, 3), name='image')
encoderTuple = encoder_build(inputs)  #https://miro.medium.com/max/1044/1*CKuZqyLSc4U8BjG3sWZHew.png
attention_inputs = encoderTuple[1]
outputs = decoder_build(encoderTuple[0]) #https://miro.medium.com/max/1044/1*CKuZqyLSc4U8BjG3sWZHew.png
model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
##model.load_weights("/workspace/work/enet_sad_naiveresize/snapshot/enetNT_Tusimple_20200424.h5", by_name=True)
#
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(1e-4)

def train_step(x,y):
    
    with tf.GradientTape() as grad:
#        encoderTuple = encoder_build(inputs)
#        distill_loss = compute_distillation(encoderTuple[1])
        img_outs     = model(x, training=False)
        model_loss   = cross_entropy(img_outs,y)
        
#        total_loss   = model_loss + distill_loss
        
    gradients    = grad.Gradient(model_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    return total_loss
    
    
train_gen = Generator(trainDataPath,2)

for epoch in range(epochs):
    
    for step in range(10):
        i,t = train_gen.__getitem__(step)
        total_loss = train_step(i,t)
        
    print(total_loss)
        
    
        
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    