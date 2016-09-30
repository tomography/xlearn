#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script
"""
# To run this example please download the test data from the classify_evaluate folder at
# http://tinyurl.com/APS-xlearn

from __future__ import print_function
import dxchange
import numpy as np
from xlearn.utils import nor_data
from xlearn.utils import extract_patches
from xlearn.utils import img_window
from xlearn.classify import model
import matplotlib.pyplot as plt
import time
import glob

np.random.seed(1337)

dim_img = 128
step = 128
patch_size = (dim_img, dim_img)
batch_size = 50
nb_classes = 2
nb_epoch = 12

# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
nb_evl = 200

start_time = time.time()
mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

mdl.load_weights('weight_center.h5')
print('The model loading time is %s seconds'%(time.time()-start_time))

Y_score = np.zeros((50, 501))
for i in range(50):
    slice_num = (i+2)*20
    datapath = '/home/oxygen/YANGX/Globus/center/test_04/slice'+str(slice_num)+'/*.tiff'
    # print(datapath)
    fnames = glob.glob(datapath)
    fnames = np.sort(fnames)
    # print(fnames)

    for j in range(len(fnames)):
        img = dxchange.read_tiff(fnames[j])
        img = -nor_data(img)
        # X_evl = np.zeros((nb_evl, dim_img, dim_img))
        # for k in range(nb_evl):
        #     X_evl[k] = img_window(img[360:1660, 440:1640], dim_img)
        X_evl = extract_patches(img[360:1660, 440:1640],
                                (128, 128), step=64, max_patches=None, random_state=None)
        X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)
        Y_evl = mdl.predict(X_evl, batch_size=batch_size)

        Y_score[i, j] = sum(np.dot(Y_evl, [0, 1]))
    # print(Y_score[i])

    #print('The evaluate score is:', Y_score[i])
    #Y_score = sum(np.round(Y_score))/len(Y_score)
    ind_max = np.argmax(Y_score[i, :])
    print('The well-centered reconstruction is:', fnames[ind_max])
    # print('The prediction runs for %s seconds'%(time.time()-start_time))
    # plt.plot(Y_score[i, :])
    # plt.show()

np.save('Y_score', Y_score)
