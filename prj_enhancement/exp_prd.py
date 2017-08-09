
import dxchange
import numpy as np
from transform import train_patch, predict_patch, train_filter, predict_filter
batch_size = 2200
nb_epoch = 40
patch_step = 1
nb_filters = 16
nb_conv = 3
patch_size = 32
patch_step = 1


imgx = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/brain_train/train_mouse_3x.tiff')
imgy = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/brain_train/train_mouse_3y.tiff')
spath = '/home/beams/YANGX/cnn_prj_enhance/tf_prd__mouse_20170420/'
wpath = 'weights/tf_mouse.h5'
mdl = train_patch(imgx, imgy, patch_size, 3, nb_filters, nb_conv, batch_size, nb_epoch)
mdl.save_weights(wpath)
img_n = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/brain_data_201610/mouse_bin.tiff')
predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)