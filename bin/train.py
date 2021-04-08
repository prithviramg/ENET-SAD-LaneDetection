# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import sys
import os
import tensorflow as tf
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import enet_sad_naiveresize.bin  # noqa: F401
    __package__ = "enet_sad_naiveresize.bin"
    
from ..models.encoder import encoder_build
from ..models.decoder import decoder_build
from ..utils.distillation import compute_distillation
from ..preprocessing.dataloader import Generator


########################## Data Input ###############################
trainDataPath = "/workspace/work/Tusimple_dataset/train_set/train.txt"
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
model.load_weights("/workspace/work/enet_sad_naiveresize/snapshot/enetNT_Tusimple_20200424.h5", by_name=True)
model.add_loss(compute_distillation(attention_inputs)) # https://arxiv.org/pdf/1908.00821.pdf - section 3.1. Self Attention Distillation

train_generator = Generator(trainDataPath,batchSize)
validation_generator = Generator(valDataPath,batchSize)
callbacks = []

callbacks.append(tf.keras.callbacks.ModelCheckpoint(
                 os.path.join(snapshotPath,
                 'enetNRSAD_Tusimple_L_{{loss:.4f}}_{ValLoss}_{{val_loss:.4f}}.h5'.format(ValLoss='VL')), 
                 monitor='val_loss', 
                 verbose=1, 
                 save_best_only=False,
                 save_weights_only=False, 
                 mode='auto', 
                 save_freq='epoch'))

callbacks.append(tf.keras.callbacks.TensorBoard(
                  log_dir=tensorboardPath, 
                  histogram_freq=2, 
                  write_graph=True, 
                  write_images=True,
                  update_freq='epoch', 
                  profile_batch=2, 
                  embeddings_freq=0,
                  embeddings_metadata=None))
                  
callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', 
                    factor=0.1, 
                    patience=15, 
                    verbose=1, 
                    mode='auto',
                    min_delta=0.0001, 
                    cooldown=0))


model.summary()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001, clipnorm=0.001), 
                     loss='binary_crossentropy',
                     metrics=['accuracy'])
model.summary()
model.fit(x = train_generator,
          epochs=epochs, 
          verbose=1, 
          callbacks=callbacks,
          validation_data=validation_generator, 
          shuffle=True)