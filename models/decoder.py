# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import tensorflow as tf


def bottleneck(encoder, output, upsample=False, reverse_module=False):
    internal = output // 4

    x = tf.keras.layers.Conv2D(internal, (1, 1), use_bias=False)(encoder)
    x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)
    if not upsample:
        x = tf.keras.layers.Conv2D(internal, (3, 3), padding='same', use_bias=True)(x)
    else:
        x = tf.keras.layers.Conv2DTranspose(filters=internal, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
    x = tf.keras.layers.PReLU(shared_axes=[1, 2])(x)

    x = tf.keras.layers.Conv2D(output, (1, 1), padding='same', use_bias=False)(x)

    other = encoder
    if encoder.get_shape()[-1] != output or upsample:
        other = tf.keras.layers.Conv2D(output, (1, 1), padding='same', use_bias=False)(other)
        other = tf.keras.layers.BatchNormalization(momentum=0.1)(other)
        if upsample and reverse_module is not False:
            other = tf.keras.layers.UpSampling2D(size=(2,2))(other)

    if upsample and reverse_module is False:
        decoder = x
    else:
        x = tf.keras.layers.BatchNormalization(momentum=0.1)(x)
        decoder = tf.keras.layers.add([x, other])
        decoder = tf.keras.layers.PReLU(shared_axes=[1, 2])(decoder)

    return decoder

# Decoder model as specified in https://arxiv.org/pdf/1606.02147.pdf
# Architecture Diagram https://miro.medium.com/max/1044/1*CKuZqyLSc4U8BjG3sWZHew.png
def decoder_build(encoder):

    enet = bottleneck(encoder, 64, upsample=True, reverse_module=True)  # bottleneck 4.0
    enet = bottleneck(enet, 64)                                         # bottleneck 4.1
    enet = bottleneck(enet, 64)                                         # bottleneck 4.2
    enet = bottleneck(enet, 16, upsample=True, reverse_module=True)     # bottleneck 5.0
    enet = bottleneck(enet, 16)                                         # bottleneck 5.1

    enet = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=(2, 2), strides=(2, 2), padding='same')(enet)
    enet = tf.keras.layers.Activation('sigmoid', name='output')(enet)
    return enet