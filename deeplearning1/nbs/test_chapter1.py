from __future__ import division,print_function

import os, json
from glob import glob
import numpy as np
np.set_printoptions(precision=4, linewidth=100)
from matplotlib import pyplot as plt




from numpy.random import random, permutation
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.models import Sequential, Model
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers import Input
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from keras.preprocessing import image

from keras.models import load_model

#from keras import backend as K
#K.set_image_dim_ordering('th')



path = "../../../data/sample/"

import utils; reload(utils)
from utils import plots

# As large as you can, but no larger than 64 is recommended. 
# If you have an older or cheaper GPU, you'll run out of memory, so will have to decrease this.
batch_size=4

# Import our class, and instantiate
import vgg16; reload(vgg16)
from vgg16 import Vgg16

model_path=path+ "ft1.h5"
vgg = Vgg16()

#print (path + "ft1.h5")

import h5py
# with h5py.File(model_path, 'a') as f:
#         if 'optimizer_weights' in f.keys():
#             del f['optimizer_weights']

# f = h5py.File(model_path)
# for k in range(f.attrs['nb_layers']):
#     if k >= len(model.layers):
#         # we don't look at the last (fully-connected) layers in the savefile
#         break
#     g = f['layer_{}'.format(k)]
#     weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
#     model.layers[k].set_weights(weights)
# f.close()

hdf5_file = h5py.File(model_path, mode='r')
print(list(hdf5_file))
vgg.model.load_weights(path+'ft1.h5')

val_batches, probs=vgg.test(path+'valid', batch_size=batch_size)

