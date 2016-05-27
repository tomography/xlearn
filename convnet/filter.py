from __future__ import print_function
import numpy as np
import scipy
#import input&output models
import time
import tomopy
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpimg
#import keras models
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Reshape, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.utils import np_utils

batch_size = 1
nb_epoch = 12
dim_img = 128
nb_filters= 32
nb_conv = 3


def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))


#read sem and txm images and extract them as output and input data
img_sem = tomopy.read_tiff('corrlated_side2/STEM1.tiff')
img_txm = tomopy.read_tiff('corrlated_side2/txm1test.tiff')
img_size = img_sem.shape

patch_size = (dim_img,dim_img)
img_input = extract_patches_2d(img_txm,patch_size,max_patches=10000)
print(img_input.shape)
img_rec = reconstruct_from_patches_2d(img_input,img_size)
imgplot = plt.imshow(img_rec,interpolation = 'none')
plt.show()

img_output = extract_patches_2d(img_sem,patch_size,max_patches=10000)
#img_input = extract_patches_2d(img_txm,patch_size)
#img_output = extract_patches_2d(img_sem,patch_size)
img_input = np.reshape(img_input,(len(img_input),1,dim_img,dim_img))
img_output = np.reshape(img_output,(len(img_input),1,dim_img,dim_img))
print('image shape',img_sem.shape)
print('input shape',img_input.shape)
print('output shape',img_output.shape)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
    border_mode='same',
    input_shape=(1, dim_img, dim_img)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(nb_filters*2, nb_conv, nb_conv, border_mode='same'))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense((dim_img/4)**2))
model.add(Reshape((1,dim_img/4,dim_img/4)))

model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(nb_filters*2,nb_conv,nb_conv,border_mode='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2,2)))
model.add(Convolution2D(1, 1, 1, border_mode='same'))



model.compile(loss='mean_squared_error',optimizer = 'Adam')
model.fit(img_input,img_output,batch_size=batch_size,nb_epoch=nb_epoch)
print('Predicting')
start = time.clock()
predicted_output = model.predict(img_input, batch_size=batch_size)
predicted_output = np.reshape(predicted_output, (len(predicted_output),dim_img,dim_img))
img_rec = reconstruct_from_patches_2d(predicted_output,img_size)
imgplot = plt.imshow(img_rec,interpolation = 'none')
plt.show()



