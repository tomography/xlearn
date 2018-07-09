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
Module containing utility routines
"""
import numpy as np
from itertools import product
import numbers
from numpy.lib.stride_tricks import as_strided

__authors__ = "Xiaogang Yang"
__copyright__ = "Copyright (c) 2018, Argonne National Laboratory"
__version__ = "0.2.0"
__docformat__ = "restructuredtext en"


def nor_data(img):
    """
    Normalize the image

    Parameters
    ----------
    img: array
        The images need to be normalized

    Returns
    -------
    img
        Description.
    """
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / (img.max() - img.min())

    return img


def mlog(img):
    img = nor_data(img)
    img = img + np.abs(img.min())
    img = img + img.mean()
    img = img /img.max()
    img = -np.log(img)

    return img

def expimg(img):
    img = nor_data(img)
    img = img + np.abs(img.min())
    img = img + img.mean()
    img = img / img.max()
    img = np.exp(img)
    return img


def rescale_intensity(img):
    img = 255*(img-img.min())/(img.max()-img.min())
    return img




def check_random_state(seed):
    """
    Turn seed into a np.random.RandomState instance
    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.

    Parameters
    ----------
    seed : type
        Description.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError('%r cannot be used to seed a numpy.random.RandomState'
                     ' instance' % seed)


def _compute_n_patches(i_h, i_w, p_h, p_w, step, max_patches=None):
    """
    Compute the number of patches that will be extracted in an image.

    Parameters
    ----------
    i_h : int
        The image height

    i_w : int
        The image with

    p_h : int
        The height of a patch

    p_w : int
        The width of a patch

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    Returns
    -------
    all_patches
        Description.
    """
    n_h = (i_h - p_h) // step + 1
    n_w = (i_w - p_w) // step + 1
    all_patches = n_h * n_w

    if max_patches:
        if (isinstance(max_patches, (numbers.Integral))
            and max_patches < all_patches):
            return max_patches
        elif (isinstance(max_patches, (numbers.Real))
              and 0 < max_patches < 1):
            return int(max_patches * all_patches)
        else:
            raise ValueError("Invalid value for max_patches: %r" % max_patches)
    else:
        return all_patches


def _extracting(arr, patch_shape=8, extraction_step=4):
    """
    Extracts patches of any n-dimensional array in place using strides.
    Given an n-dimensional array it will return a 2n-dimensional array with
    the first n dimensions indexing patch position and the last n indexing
    the patch content. This operation is immediate (O(1)). A reshape
    performed on the first n dimensions will cause numpy to copy data, leading
    to a list of extracted patches.

    Parameters
    ----------
    arr : ndarray
        n-dimensional array of which patches are to be extracted

    patch_shape : integer or tuple of length arr.ndim
        Indicates the shape of the patches to be extracted. If an
        integer is given, the shape will be a hypercube of
        sidelength given by its value.

    extraction_step : integer or tuple of length arr.ndim
        Indicates step size at which extraction shall be performed.
        If integer is given, then the step is uniform in all dimensions.

    Returns
    -------
    patches : strided ndarray
        2n-dimensional array indexing patches on first n dimensions and
        containing patches on the last n dimensions. These dimensions
        are fake, but this way no data is copied. A simple reshape invokes
        a copying operation to obtain a list of patches:
        result.reshape([-1] + list(patch_shape))
    """

    arr_ndim = arr.ndim

    if isinstance(patch_shape, numbers.Number):
        patch_shape = tuple([patch_shape] * arr_ndim)
    if isinstance(extraction_step, numbers.Number):
        extraction_step = tuple([extraction_step] * arr_ndim)

    patch_strides = arr.strides

    slices = [slice(None, None, st) for st in extraction_step]
    indexing_strides = arr[slices].strides

    patch_indices_shape = ((np.array(arr.shape) - np.array(patch_shape)) //
                           np.array(extraction_step)) + 1

    shape = tuple(list(patch_indices_shape) + list(patch_shape))
    strides = tuple(list(indexing_strides) + list(patch_strides))

    patches = as_strided(arr, shape=shape, strides=strides)
    return patches


def extract_patches(image, patch_size, step, max_patches=None, random_state=None):
    """
    Reshape a 2D image into a collection of patches
    The resulting patches are allocated in a dedicated array.

    Parameters
    ----------
    image : array, shape = (image_height, image_width) or
        (image_height, image_width, n_channels)
        The original image data. For color images, the last dimension specifies
        the channel: a RGB image would have `n_channels=3`.

    patch_size : tuple of ints (patch_height, patch_width)
        the dimensions of one patch

    step: number of pixels between two patches

    max_patches : integer or float, optional default is None
        The maximum number of patches to extract. If max_patches is a float
        between 0 and 1, it is taken to be a proportion of the total number
        of patches.

    random_state : int or RandomState
        Pseudo number generator state used for random sampling to use if
        `max_patches` is not None.

    Returns
    -------
    patches : array, shape = (n_patches, patch_height, patch_width) or
         (n_patches, patch_height, patch_width, n_channels)
         The collection of patches extracted from the image, where `n_patches`
         is either `max_patches` or the total number of patches that can be
         extracted.

    """
    i_h, i_w = image.shape[:2]
    p_h, p_w = patch_size

    if p_h > i_h:
        raise ValueError("Height of the patch should be less than the height"
                         " of the image.")

    if p_w > i_w:
        raise ValueError("Width of the patch should be less than the width"
                         " of the image.")

    # image = check_array(image, allow_nd=True)
    image = image.reshape((i_h, i_w, -1))
    n_colors = image.shape[-1]

    extracted_patches = _extracting(image, patch_shape=(p_h, p_w, n_colors), extraction_step=step)
    n_patches = _compute_n_patches(i_h, i_w, p_h, p_w, step, max_patches)
    # print n_patches
    if max_patches:
        rng = check_random_state(random_state)
        i_s = rng.randint(i_h - p_h + step, size=n_patches)
        j_s = rng.randint(i_w - p_w + step, size=n_patches)
        patches = extracted_patches[i_s, j_s, 0]
    else:
        patches = extracted_patches

    patches = patches.reshape(-1, p_h, p_w, n_colors)
    # print patches.shape
    # remove the color dimension if useless
    if patches.shape[-1] == 1:
        return patches.reshape(n_patches, p_h, p_w)
    else:
        return patches


def reconstruct_patches(patches, image_size, step):
    """
    Reconstruct the image from all of its patches.
    Patches are assumed to overlap and the image is constructed by filling in
    the patches from left to right, top to bottom, averaging the overlapping
    regions.

    Parameters
    ----------
    patches : array, shape = (n_patches, patch_height, patch_width) or
        (n_patches, patch_height, patch_width, n_channels)
        The complete set of patches. If the patches contain colour information,
        channels are indexed along the last dimension: RGB patches would
        have `n_channels=3`.

    image_size : tuple of ints (image_height, image_width) or
        (image_height, image_width, n_channels)
        the size of the image that will be reconstructed

    step: number of pixels between two patches

    Returns
    -------
    image : array, shape = image_size
        the reconstructed image
    """
    i_h, i_w = image_size[:2]
    p_h, p_w = patches.shape[1:3]
    img = np.zeros(image_size)
    # compute the dimensions of the patches array
    n_h = (i_h - p_h) // step + 1
    n_w = (i_w - p_w) // step + 1
    for p, (i, j) in zip(patches, product(range(n_h), range(n_w))):
        img[i * step:i * step + p_h, j * step:j * step + p_w] += p

    for i in range(i_h):
        for j in range(i_w):
            img[i, j] /= float(min(i + step, p_h, i_h - i) *
                               min(j + step, p_w, i_w - j))
    return img


def img_window(img, window_size):
    """
    Function Description

    Parameters
    ----------
    img : define img

    window_size : describe window_size

    Returns
    -------
    img_wd : describe img_wd

    """
    if len(img.shape) == 2:
        # img_wd = np.zeros((window_size, window_size))
        y_img, x_img = img.shape
        x_l = np.random.randint(0, x_img - window_size)
        x_r = x_l + window_size
        y_t = np.random.randint(0, y_img - window_size)
        y_b = y_t + window_size
        img_wd = img[y_t:y_b, x_l:x_r]
    else:
        img_wd = np.zeros((len(img), window_size, window_size))
        for i in range(len(img)):
            img_num, y_img, x_img = img.shape
            x_l = np.random.randint(0, x_img - window_size)
            x_r = x_l + window_size
            y_t = np.random.randint(0, y_img - window_size)
            y_b = y_t + window_size
            img_wd[i] = img[i, y_t:y_b, x_l:x_r]
    return img_wd


def extract_3d(img, patch_size, step):
    """
    Function Description

    Parameters
    ----------
    img : define img

    patch_size : describe patch_size

    step : describe step

    Returns
    -------
    patches : describe patches

    """
    # print img.shape
    # print img.dtype
    if img.ndim == 2:
        patches = extract_patches(img, patch_size, step)
    else:
        patches = extract_patches(img[0], patch_size, step)
        for i in range(len(img) - 1):
            patches_tmp = extract_patches(img[i + 1], patch_size, step)
            patches = np.concatenate((patches, patches_tmp), axis=0)
    return patches


