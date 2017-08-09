import dxchange
import numpy as np
from transform import train_patch, predict_patch, train_filter, predict_filter
batch_size = 2200
nb_epoch = 40
patch_step = 1
nb_filters = 16
nb_conv = 3
patch_size = 64
patch_step = 1


spath = '/home/beams/YANGX/cnn_prj_enhance/tf_prd_battery_20170501/'
ipath = 'weights/tf_mouse.h5'
wpath = 'weights/tf_battery.h5'

proj_start = 1200
proj_end = 1201
ind_tomo = range(proj_start, proj_end)
fname = '/home/beams1/YANGX/cnn_prj_enhance/battery1_ds/prj_00000.tiff'

#
# imgx = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/battery1_train/trainx.tif')
# imgy = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/battery1_train/trainy.tif')
#
# mdl = train_patch(imgx, imgy, patch_size, 3, nb_filters, nb_conv, batch_size, nb_epoch, ipath)
# mdl.save_weights(wpath)



img_n = dxchange.read_tiff_stack(fname, ind_tomo, digit = 5)
predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)