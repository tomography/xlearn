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

import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D

import xlearn.utils as utils


__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2018, Argonne National Laboratory"
__version__ = "0.2.0"
__docformat__ = "restructuredtext en"
__all__ = ['model',
           'train',
           'predict']


def model(dim_img, nb_filters, nb_conv):
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
    mdl.add(Convolution2D(nb_filters, (nb_conv, nb_conv),
                            padding='same', activation='relu',
                            input_shape=(dim_img, dim_img, 1)))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add(Convolution2D(nb_filters * 2, (nb_conv, nb_conv), padding='same', activation='relu'))
    mdl.add(MaxPooling2D(pool_size=(2, 2)))
    mdl.add(Convolution2D(nb_filters * 2, (nb_conv, nb_conv), padding='same', activation='relu'))

    mdl.add(Flatten())
    mdl.add(Dense(int(dim_img / 4) ** 2))
    mdl.add(Reshape((int(dim_img / 4), int(dim_img / 4), 1)))

    mdl.add(UpSampling2D(size=(2, 2)))
    mdl.add(Convolution2D(nb_filters * 2, (nb_conv, nb_conv), padding='same', activation='relu'))
    mdl.add(UpSampling2D(size=(2, 2)))
    mdl.add(Convolution2D(nb_filters, (nb_conv, nb_conv), padding='same', activation='relu'))
    mdl.add(Convolution2D(1, (1, 1), padding='same', activation='relu'))

    mdl.compile(loss='mean_squared_error', optimizer='Adam')

    return mdl


def train(img_x, img_y, patch_size, patch_step, dim_img, nb_filters, nb_conv, batch_size, nb_epoch):
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

    img_x = utils.nor_data(img_x)
    img_y = utils.nor_data(img_y)
    img_input = utils.extract_patches(img_x, patch_size, patch_step)
    img_output = utils.extract_patches(img_y, patch_size, patch_step)
    img_input = np.reshape(img_input, (img_input.shape[0], dim_img, dim_img, 1))
    img_output = np.reshape(img_output, (img_output.shape[0], dim_img, dim_img, 1))

    mdl = model(dim_img, nb_filters, nb_conv)
    print(mdl.summary())
    mdl.fit(img_input, img_output, batch_size=batch_size, epochs=nb_epoch)
    return mdl


def predict(mdl, img, patch_size, patch_step, batch_size, dim_img):
    """
    the cnn model for image transformation


    Parameters
    ----------
    img : array
        The image need to be calculated

    patch_size : (int, int)
        The patches dimension

    dim_img : int
        The input image dimension

    Returns
    -------
    img_rec
        Description.

      """
    img = np.float16(utils.nor_data(img))
    img_h, img_w = img.shape
    input_img = utils.extract_patches(img, patch_size, patch_step)
    input_img = np.reshape(input_img, (input_img.shape[0], dim_img, dim_img, 1))
    output_img = mdl.predict(input_img, batch_size=batch_size)
    del input_img
    output_img = np.reshape(output_img, (output_img.shape[0], dim_img, dim_img))
    img_rec = utils.reconstruct_patches(output_img, (img_h, img_w), patch_step)
    return img_rec
