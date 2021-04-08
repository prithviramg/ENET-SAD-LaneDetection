# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import tensorflow as tf

def compute_attention_maps(inputs,name,upsample=False):
    
    attMap = tf.reduce_sum(tf.square(inputs),axis=-1,keepdims=True,name= str(name)+"reducSum") 
    if upsample:
        attMap = tf.keras.layers.UpSampling2D(size=(2, 2), 
                                              interpolation='bilinear',
                                              name = str(name)+"bilinear")(attMap)
    attMap = tf.squeeze(attMap,axis=-1,name = str(name)+"squeeze")
    attMap = tf.reshape(attMap,
                        (tf.shape(attMap)[0],tf.shape(attMap)[1]*tf.shape(attMap)[2]),
                        name = str(name)+"reshape")
    attMap = tf.nn.softmax(attMap, 
                           axis=-1,
                           name = str(name)+"spatialSoftmax")
    return attMap

def compute_mse(x,y,name):
    
    diff = tf.math.squared_difference(x,y,name = str(name)+"squError")
    diff = tf.reduce_mean(diff,axis=0, name = str(name)+"mean")
    diff = tf.reduce_sum(diff, name = str(name)+"sum")
    return diff

# Self Attention Distillation 
# https://arxiv.org/pdf/1908.00821.pdf - section 3.1. Self Attention Distillation
def compute_distillation(attention_inputs):

    inp1,inp2,inp3,inp4 = attention_inputs 
    
    attMap1          = compute_attention_maps(inp1,"attmap1_")
    attMap2_upsample = compute_attention_maps(inp2,"attmap2UP_",upsample=True)
    attMap2          = compute_attention_maps(inp2,"attmap2_")
    attMap3_upsample = compute_attention_maps(inp3,"attmap3UP_",upsample=True)
    attMap3          = compute_attention_maps(inp3,"attmap3_")
    attMap4          = compute_attention_maps(inp4,"attmap4_")
    
    distillation1 = compute_mse(attMap1,attMap2_upsample,"distil1_")
    distillation2 = compute_mse(attMap2,attMap3_upsample,"distil2_")
    distillation3 = compute_mse(attMap3,attMap4,"distil3_")
    
    return tf.math.add_n([distillation1,distillation2,distillation3], name="distill_loss")