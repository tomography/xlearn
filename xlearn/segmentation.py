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
Module containing model_choose, seg_train and seg_predict routines
"""
import numpy as np
import tensorflow as tf
import time
import dxchange
from xlearn.utils import nor_data, extract_3d, reconstruct_patches
from xlearn.models import transformer2, transformer3_pooling

__authors__ = "Xiaogang Yang, Francesco De Carlo"
__copyright__ = "Copyright (c) 2018, Argonne National Laboratory & DESY"
__version__ = "0.2.1"
__docformat__ = "restructuredtext en"
__all__ = ['model_choose',
           'seg_train',
           'seg_predict']

def _get_seg_kwargs():
    return {
        'patch_size': 32,
        'patch_step': 1,
        'conv_nb': 32,
        'conv_size': 3,
        'batch_size': 1000,
        'epoch_nb': 20,
        'model_layers': 2,
        'lr': 1e-3,
        'nor_slice': False
    }

def seg_train(img_x, img_y, **kwargs):
    """
    Function description.

    Parameters
    ----------
    img_x: array, 2D or 3D
        Training input of the model. It is the raw image for the segmentation.

    img_y: array, 2D or 3D
        Training output of the model. It is the corresponding segmentation of the training input.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    conv_nb: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    conv_size: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    model_layers: int
          Number of the downsampling for the images in the model.

    gpu_nb: int
          Number of GPUs you want to use for the training.


    Returns
    -------
    mdl
        The trained CNN model for segmenation. The model can be saved for future segmentations.
    """
    seg_kwargs = ['patch_size', 'patch_step', 'conv_nb', 'conv_size', 'batch_size',
                  'epoch_nb', 'model_layers', 'lr']
    kwargs_defaults = _get_seg_kwargs()
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    for kw in seg_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    patch_shape = (kwargs['patch_size'], kwargs['patch_size'])
    # print img_x.shape
    # print img_x.max(), img_x.min()
    img_x = nor_data(img_x)
    img_y = nor_data(img_y)
    # print(img_x.shape)
    # print(img_x.max(), img_x.min())
    # print(img_y.max(), img_y.min())


    train_x = extract_3d(img_x, patch_shape, kwargs['patch_step'])
    train_y = extract_3d(img_y, patch_shape, kwargs['patch_step'])
    # print train_x.shape
    # print train_x.max(), train_x.min()
    train_x = np.reshape(train_x, (len(train_x), kwargs['patch_size'], kwargs['patch_size'], 1))
    train_y = np.reshape(train_y, (len(train_y), kwargs['patch_size'], kwargs['patch_size'], 1))
    with strategy.scope():
        mdl = model_choose(kwargs['patch_size'], kwargs['patch_size'], kwargs['conv_nb'], kwargs['conv_size'],
                           kwargs['model_layers'], kwargs['lr'])


    # print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=kwargs['batch_size'], epochs=kwargs['epoch_nb'])
    return mdl

def seg_predict(img, wpath, spath, **kwargs):
    """
    Function description

    Parameters
    ----------
    img : array
        The images need to be segmented.

    wpath: string
        The path where the trained weights of the model can be read.

    spath: string
        The path to save the segmented images.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    conv_nb: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    conv_size: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    model_layers: int
          Number of the downsampling for the images in the model.

    gpu_nb: int
          Number of GPUs you want to use for the training.

    Returns
    -------
    save the segmented images to the spath.

      """
    seg_kwargs = ['patch_size', 'patch_step', 'conv_nb', 'conv_size', 'batch_size',
                  'epoch_nb', 'model_layers', 'lr', 'nor_slice']
    kwargs_defaults = _get_seg_kwargs()
    for kw in seg_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    patch_shape = (kwargs['patch_size'], kwargs['patch_size'])
    img = np.float32(nor_data(img))
    mdl = model_choose(kwargs['patch_size'], kwargs['patch_size'], kwargs['conv_nb'],
                       kwargs['conv_size'], kwargs['model_layers'], kwargs['lr'])
    # with strategy.scope():
    #     mdl = model_choose(kwargs['patch_size'], kwargs['patch_size'], kwargs['conv_nb'],
    #                        kwargs['conv_size'], kwargs['model_layers'], kwargs['gpu_nb'])

    # print(mdl.summary())
    mdl.load_weights(wpath)
    if img.ndim == 2:
        ih, iw = img.shape
        predict_y = pred_single(mdl, img, ih, iw, patch_shape, kwargs['patch_step'],
                                kwargs['patch_size'], kwargs['batch_size'])
        fname = spath+'seg'
        dxchange.write_tiff(predict_y, fname, dtype='float32')
    else:
        pn, ih, iw = img.shape
        for i in range(pn):
            print('Processing the %s th image' % i)
            tstart = time.time()
            predict_x = img[i]
            if nor_slice:
                predict_x=nor_data(predict_x)
            predict_y = pred_single(mdl, predict_x, ih, iw, patch_shape, kwargs['patch_step'],
                                    kwargs['patch_size'], kwargs['batch_size'])
            predict_y = np.float32(predict_y)
            fname = spath+'seg' + "-%03d" % (i)
            dxchange.write_tiff(predict_y, fname, dtype='float32')
            print('The prediction runs for %s seconds' % (time.time() - tstart))

def model_choose(ih, iw, conv_nb, conv_size, model_layers, lr):
    if model_layers == 3:
        mdl = transformer3_pooling(ih, iw, conv_nb, conv_size, lr)
    else:
        mdl = transformer2(ih, iw, conv_nb, conv_size, lr)
    return mdl

def pred_single(mdl, predict_x, ih, iw, patch_shape, patch_step, patch_size, batch_size):
    predict_x = extract_3d(predict_x, patch_shape, patch_step)
    predict_x = np.reshape(predict_x, (predict_x.shape[0], patch_size, patch_size, 1))
    predict_y = mdl.predict(predict_x, batch_size=batch_size)
    predict_y = np.reshape(predict_y, (predict_y.shape[0], patch_size, patch_size))
    predict_y = reconstruct_patches(predict_y, (ih, iw), patch_step)
    return predict_y


