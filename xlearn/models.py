#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015-2018, UChicago Argonne, LLC. All rights reserved.    #
#                                                                         #
# Copyright 2018. UChicago Argonne, LLC. This software was produced       #
# under U.S. Government contract DE-AC02-06CH11357 for Argonne National   #
# Laboratory (ANL), which is operated by UChicago Argonne, LLC for the    #
# U.S. Department of Energy. The U.S. Government has rights to use,       #
# reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR    #
# UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR        #
# ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is     #
# modified to produce derivative works, such modified software should     #
# be clearly marked, so as not to confuse it with the version available   #
# from ANL.                                                               #
#                                                                         #
# Additionally, redistribution and use in source and binary forms, with   #
# or without modification, are permitted provided that the following      #
# conditions are met:                                                     #
#                                                                         #
#     * Redistributions of source code must retain the above copyright    #
#       notice, this list of conditions and the following disclaimer.     #
#                                                                         #
#     * Redistributions in binary form must reproduce the above copyright #
#       notice, this list of conditions and the following disclaimer in   #
#       the documentation and/or other materials provided with the        #
#       distribution.                                                     #
#                                                                         #
#     * Neither the name of UChicago Argonne, LLC, Argonne National       #
#       Laboratory, ANL, the U.S. Government, nor the names of its        #
#       contributors may be used to endorse or promote products derived   #
#       from this software without specific prior written permission.     #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago     #
# Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,        #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;        #
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER        #
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT      #
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN       #
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE         #
# POSSIBILITY OF SUCH DAMAGE.                                             #
# #########################################################################

"""
Module containing model, predict and train routines
"""

from __future__ import print_function
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Dense, Reshape, Flatten, \
    Dropout, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Activation
from tensorflow.python.keras.utils import multi_gpu_model

# from keras.models import Sequential, Model
# from keras.layers import Dense, Reshape, Flatten, \
#     Dropout, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Activation
# from keras.utils import multi_gpu_model

__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, Argonne National Laboratory"
__version__ = "0.2.0"
__docformat__ = "restructuredtext en"
__all__ = ['classifier',
           'transformer2',
           'transformer3_super',
           'transformer3_pooling',
           'transformer3_direct',
           'transformer3_filter']

def classifier(ih, iw, nb_conv, size_conv, nb_classes):
    """
    The cnn model for image classification

    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel 
    Returns
    -------
    mdl 
        Description.

    """

    mdl = Sequential()

    mdl.add(Conv2D(nb_conv, size_conv, size_conv,
                            border_mode='valid',
                            input_shape=(ih, iw, 1)))
    mdl.add(Activation('relu'))
    mdl.add(Conv2D(nb_conv, size_conv, size_conv))
    mdl.add(Activation('relu'))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add(Dropout(0.25))

    mdl.add(Conv2D(nb_conv * 2, size_conv, size_conv))
    mdl.add(Activation('relu'))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add(Dropout(0.25))

    mdl.add(Flatten())
    mdl.add(Dense(128))
    mdl.add(Activation('relu'))
    mdl.add(Dropout(0.5))
    mdl.add(Dense(nb_classes))
    mdl.add(Activation('softmax'))

    mdl.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])

    return mdl


def transformer2(ih, iw, nb_conv, size_conv, nb_gpu = 1):
    """
    The simple cnn model for image transformation with 2 times of downsampling. It is a choice for fast running.
    However, it will lose resolution during the transformation.

    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel
    Returns
    -------
    mdl
        Description.
    """
    inputs = Input((ih, iw, 1))

    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(pool2)
    #
    fc1 = Flatten()(conv3)
    fc1 = Dense(iw * ih / 16)(fc1)
    fc1 = Reshape((ih // 4, iw // 4, 1))(fc1)

    conv4 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(fc1)

    up1 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv2], axis=3)

    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv1], axis=3)

    conv7 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv7)

    conv8 = Conv2DTranspose(1, (size_conv, size_conv), activation='relu', padding='same')(conv7)

    mdl = Model(inputs=inputs, outputs=conv8)
    if nb_gpu > 1:
        mdl = multi_gpu_model(mdl, nb_gpu)

    mdl.compile(loss='mse', optimizer='Adam')

    return mdl

def transformer3_pooling(ih, iw, nb_conv, size_conv, nb_gpu):

    """
    The cnn image transformation model with 3 times of downsampling. The downsampling uses maxpooling.

    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel
    Returns
    -------
    mdl
        Description.

    """

    inputs = Input((ih, iw, 1))

    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(1, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    #
    fc1 = Flatten()(conv4)
    fc1 = Dense(iw * ih / 128, activation='relu')(fc1)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Dense(iw * ih / 128, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Dense(iw * ih / 64, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Reshape((int(ih // 8), int(iw // 8), 1))(fc1)

    fc2 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(fc1)
    fc2 = Conv2DTranspose(nb_conv * 8, (size_conv, size_conv), activation='relu', padding='same')(fc2)

    up1 = concatenate([UpSampling2D(size=(2, 2))(fc2), conv3], axis=3)
    
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv6)

    up2 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv2], axis=3)

    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv7)

    up3 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv1], axis=3)

    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(up3)
    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv8)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)

    mdl = Model(inputs=inputs, outputs=conv8)
    if nb_gpu > 1:
        mdl = multi_gpu_model(mdl, nb_gpu)

    mdl.compile(loss='mse', optimizer='Adam')
    return mdl


def transformer3_super(ih, iw, nb_conv, size_conv):

    """
    The cnn model for image transformation with 3 times downsampling. The downsampling uses strides. The model also
    merge the convolution layers from encoding and decoding parts to keep the resolution of the image. It works good
    for super-resolution and image enhancement.


    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel
    Returns
    -------
    mdl
        Description.

    """

    inputs = Input((ih, iw, 1))

    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(inputs)
    conv1a = Conv2D(nb_conv, (size_conv, size_conv), strides=(2, 2), activation='relu', padding='same')(conv1)

    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv1a)
    conv2a = Conv2D(nb_conv * 2, (size_conv, size_conv), strides=(2, 2), activation='relu', padding='same')(conv2)

    conv3 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv2a)
    conv3a = Conv2D(nb_conv * 2, (size_conv, size_conv), strides=(2, 2), activation='relu', padding='same')(conv3)

    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv3a)
    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(1, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    #
    fc1 = Flatten()(conv4)
    fc1 = Dense(iw * ih / 128, activation='relu')(fc1)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Dense(iw * ih / 128, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Dense(iw * ih / 64, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Reshape((ih // 8, iw // 8, 1))(fc1)

    fc2 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(fc1)
    fc2 = Conv2DTranspose(nb_conv * 8, (size_conv, size_conv), trides=(2, 2), activation='relu', padding='same')(fc2)
    
    up1 = concatenate([fc2, conv3], axis=3)

    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), strides=(2, 2), 
                            activation='relu', padding='same')(conv6)

    up2 = concatenate([conv6, conv2], axis=3)

    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv7)

    up3 = concatenate([conv7, conv1], axis=3)

    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(up3)
    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv8)
    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)

    mdl = Model(inputs=inputs, outputs=conv8)
    mdl.compile(loss=psnr, optimizer='Adam', metrics=['mse'])
    return mdl


def transformer3_direct(ih, iw, nb_conv, size_conv):
    """
    The cnn model for image transformation with 3 times downsampling. The downsampling uses strides. It does not have
    merged layers. It will lose resolution but possible to generate more different images.

    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel
    Returns
    -------
    mdl
        Description.

    """

    inputs = Input((ih, iw, 1))

    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(inputs)
    conv1a = Conv2D(nb_conv, (size_conv, size_conv),
                    strides=(2, 2), activation='relu', padding='same')(conv1)

    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv1a)
    conv2a = Conv2D(nb_conv * 2, (size_conv, size_conv),
                    strides=(2, 2), activation='relu', padding='same')(conv2)

    conv3 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv2a)
    conv3 = Conv2D(nb_conv * 4, (size_conv, size_conv),
                    strides=(2, 2), activation='relu', padding='same')(conv3)

    fc1 = Flatten()(conv3)
    fc1 = Dense(iw * ih / 64, activation='relu')(fc1)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Dense(iw * ih / 16, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Reshape((ih // 4, iw // 4, 1))(fc1)

    fc2 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(fc1)
    fc2 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv),
                          strides=(2, 2), activation='relu', padding='same')(fc2)

    up1 = concatenate([fc2, conv2], axis=3)

    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv6)

    up2 = concatenate([conv6, conv1], axis=3)

    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv7)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv7)

    mdl = Model(inputs=inputs, outputs=conv8)
    mdl.compile(loss=psnr, optimizer='Adam', metrics=['mse'])
    return mdl


def transformer3_filter(ih, iw, nb_conv, size_conv):

    """
    The cnn model for image transformation with 3 times downsampling. This model does not include fully connected
    layers.

    Parameters
    ----------
    ih, iw : int
        The input image dimension

    nb_conv : int
        Number of convolution kernels for each layer

    size_conv : int
        The size of convolution kernel
    Returns
    -------
    mdl
        Description.

    """

    inputs = Input((ih, iw, 1))

    conv1 = Conv2D(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(inputs)
    conv1a = Conv2D(nb_conv, (size_conv, size_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv1)

    conv2 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv1a)
    conv2a = Conv2D(nb_conv * 2, (size_conv, size_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv2)

    conv3 = Conv2D(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(conv2a)
    conv3a = Conv2D(nb_conv * 2, (size_conv, size_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv3)

    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv3a)
    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    #
    conv5 = Conv2DTranspose(nb_conv * 4, (size_conv, size_conv), activation='relu', padding='same')(conv4)
    conv5 = Conv2DTranspose(nb_conv * 8, (size_conv, size_conv),
                           strides=(2, 2), activation='relu', padding='same')(conv5)

    up1 = concatenate([conv5, conv3],  axis=3)

    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv6)

    up2 = concatenate([conv6, conv2], axis=3)

    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_conv * 2, (size_conv, size_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv7)

    up3 = concatenate([conv7, conv1], axis=3)

    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(up3)
    conv8 = Conv2DTranspose(nb_conv, (size_conv, size_conv), activation='relu', padding='same')(conv8)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)

    mdl = Model(inputs=inputs, outputs=conv8)
    mdl.compile(loss= 'mse', optimizer='Adam', metrics=['accuracy'])
    return mdl

def psnr(y_true, y_pred):
    """
    The cost function by computing the psnr.
    """
    return 1/(10.0 * tf.log(1.0 / (tf.mean(tf.square(y_pred - y_true)))) / tf.log(10.0))