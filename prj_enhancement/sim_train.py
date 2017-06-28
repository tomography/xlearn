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

imgx = np.zeros((2, 512, 728))
imgy = np.zeros((2, 512, 728))

img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_05.tiff')
img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
imgx[0]= img_n[0]
imgx[1]= img_n[180]
imgy[0]= img_o[0]
imgy[1]= img_o[180]
spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n05/'
wpath = 'weights/tf_05.h5'
mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
mdl.save_weights(wpath)
predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)


# img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_10.tiff')
# img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
# imgx[0]= img_n[0]
# imgx[1]= img_n[180]
# imgy[0]= img_o[0]
# imgy[1]= img_o[180]
# spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n10/'
# wpath = 'weights/tf_10.h5'
# mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
# mdl.save_weights(wpath)
# predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
# img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_15.tiff')
# img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
# imgx[0]= img_n[0]
# imgx[1]= img_n[180]
# imgy[0]= img_o[0]
# imgy[1]= img_o[180]
# spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n15/'
# wpath = 'weights/tf_15.h5'
# mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
# mdl.save_weights(wpath)
# predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
#
# img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_20.tiff')
# img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
# imgx[0]= img_n[0]
# imgx[1]= img_n[180]
# imgy[0]= img_o[0]
# imgy[1]= img_o[180]
# spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n20/'
# wpath = 'weights/tf_20.h5'
# mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
# mdl.save_weights(wpath)
# predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
# img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_25.tiff')
# img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
# imgx[0]= img_n[0]
# imgx[1]= img_n[180]
# imgy[0]= img_o[0]
# imgy[1]= img_o[180]
# spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n25/'
# wpath = 'weights/tf_25.h5'
# mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
# mdl.save_weights(wpath)
# predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
# img_n= dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_30.tiff')
# img_o = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/sim/sim_prj2.tiff')
# imgx[0]= img_n[0]
# imgx[1]= img_n[180]
# imgy[0]= img_o[0]
# imgy[1]= img_o[180]
# spath = '/home/beams/YANGX/cnn_prj_enhance/sim/tf_n30/'
# wpath = 'weights/tf_30.h5'
# mdl = train_patch(imgx, imgy, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch)
# mdl.save_weights(wpath)
# predict_patch(img_n, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)