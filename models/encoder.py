# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import tensorflow as tf
from ..preprocessing.dataloader import Generator

def initial_block(inputs, filters=13, kernel=(3,3), strides=(2, 2)):
    conv = tf.keras.layers.Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    max_pool = tf.keras.layers.MaxPooling2D()(inputs)
    merged = tf.keras.layers.concatenate([conv, max_pool], axis=3)
    return merged # total 16 layers

def bottleneck(inputs, outputs, internal_scale=4, asymmetric=0, dilated=0, downsample=False, dropout_rate=0.1):
    # main branch
    internal = outputs // internal_scale
    encoder = inputs

    # 1x1
    input_stride = 2 if downsample else 1  # the 1st 1x1 projection is replaced with a 2x2 convolution when downsampling
    encoder = tf.keras.layers.Conv2D(internal, (input_stride, input_stride),
                     # padding='same',
                     strides=(input_stride, input_stride), use_bias=False)(encoder)
    # Batch normalization + PReLU
    encoder = tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)

    # conv
    if not asymmetric and not dilated:
        encoder = tf.keras.layers.Conv2D(internal, (3, 3), padding='same')(encoder)
    elif asymmetric:
        encoder = tf.keras.layers.Conv2D(internal, (1, asymmetric), padding='same', use_bias=False)(encoder)
        encoder = tf.keras.layers.Conv2D(internal, (asymmetric, 1), padding='same')(encoder)
    elif dilated:
        encoder = tf.keras.layers.Conv2D(internal, (3, 3), dilation_rate=(dilated, dilated), padding='same')(encoder)
    else:
        raise(Exception('You shouldn\'t be here'))

    encoder = tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
    
    # 1x1
    encoder = tf.keras.layers.Conv2D(outputs, (1, 1), use_bias=False)(encoder)

    encoder = tf.keras.layers.BatchNormalization(momentum=0.1)(encoder)  # enet_unpooling uses momentum of 0.1, keras default is 0.99
    encoder = tf.keras.layers.SpatialDropout2D(dropout_rate)(encoder)

    other = inputs
    # other branch
    if downsample:
        other = tf.keras.layers.MaxPooling2D()(other)

        other = tf.keras.layers.Permute((1, 3, 2))(other)
        pad_feature_maps = outputs - inputs.get_shape().as_list()[3]
        tb_pad = (0, 0)
        lr_pad = (0, pad_feature_maps)
        other = tf.keras.layers.ZeroPadding2D(padding=(tb_pad, lr_pad))(other)
        other = tf.keras.layers.Permute((1, 3, 2))(other)

    encoder = tf.keras.layers.add([encoder, other])
    encoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(encoder)
    if downsample:
        return encoder
    else:
        return encoder
  

# Encoder model as specified in https://arxiv.org/pdf/1606.02147.pdf
# Architecture Diagram https://miro.medium.com/max/1044/1*CKuZqyLSc4U8BjG3sWZHew.png
def encoder_build(inputs, dropout_rate=0.01):
    attention_inputs = ()
    enet = initial_block(inputs)                  
    enet = tf.keras.layers.BatchNormalization(momentum=0.1)(enet)           # enet_unpooling uses momentum of 0.1, keras default is 0.99
    enet = tf.keras.layers.PReLU(shared_axes=[1, 2])(enet)                  # initial block
    
    #######################################################################################
    attention_inputs = attention_inputs + (enet,)                           # Attention Tensor 1
    #######################################################################################

    enet = bottleneck(enet, 64, downsample=True, dropout_rate=dropout_rate) # bottleneck 1.0

    enet = bottleneck(enet, 64, dropout_rate=dropout_rate)                  # bottleneck 1.1
    enet = bottleneck(enet, 64, dropout_rate=dropout_rate)                  # bottleneck 1.2
    enet = bottleneck(enet, 64, dropout_rate=dropout_rate)                  # bottleneck 1.3
    enet = bottleneck(enet, 64, dropout_rate=dropout_rate)                  # bottleneck 1.4
    
    #######################################################################################
    attention_inputs = attention_inputs + (enet,)                           # Attention Tensor 2
    #######################################################################################
    
    enet = bottleneck(enet, 128, downsample=True)                           # bottleneck 2.0
    
    # bottleneck 2.x and 3.x
    enet = bottleneck(enet, 128)                                            # bottleneck 2.1
    enet = bottleneck(enet, 128, dilated=2)                                 # bottleneck 2.2
    enet = bottleneck(enet, 128, asymmetric=5)                              # bottleneck 2.3
    enet = bottleneck(enet, 128, dilated=4)                                 # bottleneck 2.4
    enet = bottleneck(enet, 128)                                            # bottleneck 2.5
    enet = bottleneck(enet, 128, dilated=8)                                 # bottleneck 2.6
    enet = bottleneck(enet, 128, asymmetric=5)                              # bottleneck 2.7
    enet = bottleneck(enet, 128, dilated=16)                                # bottleneck 2.8
    
    #######################################################################################
    attention_inputs = attention_inputs + (enet,)                           # Attention Tensor 3
    #######################################################################################
    
    enet = bottleneck(enet, 128)                                            # bottleneck 3.1
    enet = bottleneck(enet, 128, dilated=2)                                 # bottleneck 3.2
    enet = bottleneck(enet, 128, asymmetric=5)                              # bottleneck 3.3
    enet = bottleneck(enet, 128, dilated=4)                                 # bottleneck 3.4
    enet = bottleneck(enet, 128)                                            # bottleneck 3.5
    enet = bottleneck(enet, 128, dilated=8)                                 # bottleneck 3.6
    enet = bottleneck(enet, 128, asymmetric=5)                              # bottleneck 3.7
    enet = bottleneck(enet, 128, dilated=16)                                # bottleneck 3.8
    
    #######################################################################################
    attention_inputs = attention_inputs + (enet,)                           # Attention Tensor 4
    #######################################################################################

    return enet, attention_inputs
