#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script 
"""

from __future__ import print_function
from xlearn.transform import train
import dxchange


batch_size = 800
nb_epoch = 10
dim_img = 20
nb_filters = 32
nb_conv = 3
patch_step = 4

patch_size = (dim_img, dim_img)

# define the data path
spath = '../../test/test_data/'

# read the training data
img_x = dxchange.read_tiff(spath + 'training_input.tiff')
img_y = dxchange.read_tiff(spath + 'training_output.tiff')

# train and save the model
model = train(img_x, img_y, patch_size, patch_step, dim_img, nb_filters, nb_conv, batch_size, nb_epoch)
model.save_weights('transform_training_weights.h5')

