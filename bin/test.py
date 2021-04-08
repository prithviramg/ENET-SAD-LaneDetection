# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 13:51:39 2020

@author: prith
"""
import sys
import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import enet_sad_naiveresize.bin  # noqa: F401
    __package__ = "enet_sad_naiveresize.bin"

from ..models.encoder import encoder_build
from ..models.decoder import decoder_build

#imgPath = "/workspace/work/LADYBUG_DATASET/inputs/21022020/im_1567006779_310.png"
imgPath = imgPath.replace("\\","/")

model = tf.keras.models.load_model("/workspace/work/enet_sad_naiveresize/snapshot/enetNRSAD_Tusimple_L_0.0801_VL_0.0856.h5")
#model.summary()

img = np.asarray(Image.open(imgPath).convert('RGB'))
img1 = img.copy()

img = img.astype(np.float32)
img /= 127.5
img -= 1
img = np.expand_dims(img,axis = 0)

output = model.predict(img)
out = np.concatenate((output[0],np.zeros_like(output[0]),np.zeros_like(output[0])),axis=-1)
out *= 255
out = np.array(out,dtype=np.uint8)

img2 = cv2.addWeighted(img1, 1, out, 1, 0) 

plt.imsave("/workspace/work/enet_sad_naiveresize/predicitions/16_enet.jpg",img2)
plt.imsave("/workspace/work/enet_sad_naiveresize/predicitions/16_mask_enet.jpg",out)