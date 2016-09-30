#!/usr/bin/env python
# -*- coding: utf-8 -*-

# #########################################################################
# Copyright (c) 2015, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Copyright 2015. UChicago Argonne, LLC. This software was produced       #
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

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import xlearn.utils as utils


__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2016, Argonne National Laboratory"
__version__ = "0.1.0"
__docformat__ = "restructuredtext en"
__all__ = ['model',
           'train']


def model(dim_img, nb_filters, nb_conv, nb_classes):
    """
    the cnn model for image transformation

    Parameters
    ----------
    dim_img : int
        The input image dimension

    nb_filters : int
        Number of filters

    nb_conv : int
        The convolution weight dimension

    Returns
    -------
    mdl
        Description.

    """

    mdl = Sequential()

    mdl.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='valid',
                            input_shape=(1, dim_img, dim_img)))
    mdl.add(Activation('relu'))
    mdl.add(Convolution2D(nb_filters, nb_conv, nb_conv))
    mdl.add(Activation('relu'))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add(Dropout(0.25))

    mdl.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv))
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

def train(x_train, y_train, x_test, y_test, dim_img, nb_filters, nb_conv, batch_size, nb_epoch, nb_classes):
    """
    Function description.

    Parameters
    ----------
    parameter_01 : type
        Description.

    parameter_02 : type
        Description.

    parameter_03 : type
        Description.

    Returns
    -------
    return_01
        Description.
    """
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

    mdl.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(x_test, y_test))
    mdl.save_weights('weight_center.h5')
    score = mdl.evaluate(x_test, y_test, show_accuracy=True, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return mdl

