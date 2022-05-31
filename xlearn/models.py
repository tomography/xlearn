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
from tensorflow.keras import layers, Input, optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, \
    Dropout, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Activation

# from tensorflow.python.keras.utils import multi_gpu_model

# from keras.models import Sequential, Model
# from keras.layers import Dense, Reshape, Flatten, \
#     Dropout, Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Activation
# from keras.utils import multi_gpu_model

__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, Argonne National Laboratory"
__version__ = "0.3.0"
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


def transformer2(ih, iw, nb_conv, size_conv, lr):
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
    # if nb_gpu > 1:
    #     mdl = multi_gpu_model(mdl, nb_gpu)
    opt = optimizers.Adam(learning_rate=lr)
    mdl.compile(loss='mse', optimizer=opt)

    return mdl

def transformer3_pooling(ih, iw, nb_conv, size_conv, lr):

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
    # if nb_gpu > 1:
    #     mdl = multi_gpu_model(mdl, nb_gpu)

    opt = optimizers.Adam(learning_rate=lr)
    mdl.compile(loss='mse', optimizer=opt)
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


def dense_norm(units, dropout, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Dense(units, activation=tf.nn.tanh, use_bias=True, kernel_initializer=initializer))
    result.add(layers.Dropout(dropout))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    #     result.add(layers.LeakyReLU())

    return result


def conv2d_norm(filters, size, strides, apply_batchnorm=True):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2D(filters, size, strides=strides, padding='same',
                      kernel_initializer=initializer, activation=tf.nn.elu))

    if apply_batchnorm:
        result.add(layers.BatchNormalization())

    # result.add(layers.LeakyReLU())

    return result


def dconv2d_norm(filters, size, strides, apply_dropout=False):
    initializer = tf.random_normal_initializer()

    result = tf.keras.Sequential()
    result.add(
        layers.Conv2DTranspose(filters, size, strides=strides,
                               padding='same',
                               kernel_initializer=initializer,
                               use_bias=False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result


def make_generator(img_h, img_w, conv_num, conv_size, dropout, output_num):
    units = 128
    fc_size = img_w ** 2
    inputs = Input(shape=(img_h, img_w, 1))
    x = tf.keras.layers.Flatten()(inputs)
    fc_stack = [
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(units, dropout),
        dense_norm(fc_size, 0),
    ]

    conv_stack = [
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size+2, 1),
        conv2d_norm(conv_num, conv_size, 1),

    ]

    dconv_stack = [
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size+2, 1),
        dconv2d_norm(conv_num, conv_size, 1),
    ]

    last = conv2d_norm(output_num, 3, 1)

    for fc in fc_stack:
        x = fc(x)

    x = tf.reshape(x, shape=[-1, img_w, img_w, 1])
    # Convolutions
    for conv in conv_stack:
        x = conv(x)

    for dconv in dconv_stack:
        x = dconv(x)
    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_filter(img_h, img_w):
    inputs = Input(shape=[img_h, img_w, 1])
    down_stack = [
        conv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 64)
        conv2d_norm(16, 3, 1),
        # conv2d_norm(16, 3, 1),
        # conv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 128)
        # conv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 256)
        # conv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 8, 8, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 4, 4, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 2, 2, 512)
        # conv2d_norm(512, 4, 2),  # (batch_size, 1, 1, 512)
    ]

    up_stack = [
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 2, 2, 1024)
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 4, 4, 1024)
        # dconv2d_norm(512, 4, 2, apply_dropout=True),  # (batch_size, 8, 8, 1024)
        # dconv2d_norm(512, 4, 2),  # (batch_size, 16, 16, 1024)
        # dconv2d_norm(256, 4, 2),  # (batch_size, 32, 32, 512)
        # dconv2d_norm(128, 4, 2),  # (batch_size, 64, 64, 256)
        dconv2d_norm(16, 3, 1),  # (batch_size, 128, 128, 128)
        dconv2d_norm(16, 3, 1)
    ]
    last = conv2d_norm(1, 3, 1)
    # initializer = tf.random_normal_initializer(0., 0.02)
    # last = tf.keras.layers.Conv2DTranspose(1, 3,
    #                                        strides=1,
    #                                        padding='same',
    #                                        kernel_initializer=initializer,
    #                                        activation='tanh')  # (batch_size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        # x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def make_discriminator(nang, px):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                            input_shape=[nang, px, 1]))
    model.add(layers.Conv2D(16, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
    # model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
