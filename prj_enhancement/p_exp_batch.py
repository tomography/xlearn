import dxchange
import numpy as np
import tomopy
from skimage.transform import downscale_local_mean
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.DEBUG)
import glob
import os
from transform import train_patch, predict_patch, train_filter, predict_filter

batch_size = 2200
nb_epoch = 40
patch_step = 1
nb_filters = 16
nb_conv = 3
patch_size = 32
patch_step = 1


sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

def ds(img):
    il, ih, iw = img.shape
    img_tmp = np.zeros((il, ih / 2, iw / 2))
    for i in range(len(img)):
        img_tmp[i] = downscale_local_mean(img[i], (2, 2))
    return img_tmp
fnames = glob.glob('/home/xiyang/data_20170608/')
wpath = 'weights/200ms_all'

for i in range(len(wpath)):
    fname = fnames[i]
    spath = '/home/xiyang/data_20170608/'+fname[:-3]+'/'
    if not os.path.exists(spath):
        os.makedirs(spath)
    data = dxchange.read_hdf5(fname,'/exchange/data')
    dark = dxchange.read_hdf5(fname,'/exchange/data_dark')
    white = dxchange.read_hdf5(fname, '/exchange/data_white')
    print data.shape, dark.shape, white.shape
    print data.min(), data.max()
    data = tomopy.normalize(data, white, dark)
    print data.min(), data.max()
    print data.shape
    data[data>10] = 0.5
    data[data<-10] = 0
    data = ds(data)
    predict_patch(data, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)


