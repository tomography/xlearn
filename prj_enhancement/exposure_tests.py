import dxchange
import numpy as np
from transform import train_patch, predict_patch, train_filter, predict_filter
import tomopy
from skimage.transform import downscale_local_mean

def ds(img):
    il, ih, iw = img.shape
    img_tmp = np.zeros((il, ih / 2, iw / 2))
    for i in range(len(img)):
        img_tmp[i] = downscale_local_mean(img[i], (2, 2))
    return img_tmp

batch_size = 2200
nb_epoch = 20
patch_step = 1
nb_filters = 16
nb_conv = 3
patch_size = 32
patch_step = 1


# ipath = 'weights/tf_mouse.h5'
spath = '/home/beams/YANGX/cnn_prj_enhance/exposure_tests/c3s_test1/'

imgx = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/trainx_400ms.tiff')
# imgx2 = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/train2.tiff')
# imgx3 = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/train3.tiff')
# imgx4 = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/train4.tiff')
# imgx5 = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/train5.tiff')
imgy = dxchange.read_tiff('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/trainy.tiff')

# mdl = train_patch(imgx1[0],imgy, patch_size, 2, nb_filters, nb_conv, batch_size, nb_epoch, ipath)
# wpath = 'weights/100ms.h5'
# mdl.save_weights(wpath)
# predict_patch(imgx1, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
# mdl = train_patch(imgx2[0], imgy, patch_size, 2, nb_filters, nb_conv, batch_size, nb_epoch, ipath)
# wpath = 'weights/200ms.h5'
# mdl.save_weights(wpath)
# predict_patch(imgx2, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
#
# mdl = train_patch(imgx3[0], imgy, patch_size, 2, nb_filters, nb_conv, batch_size, nb_epoch, ipath)
# wpath = 'weights/300ms.h5'
# mdl.save_weights(wpath)
# predict_patch(imgx3, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)
print imgx.shape, imgy.shape
mdl = train_patch(imgx, imgy, patch_size, 2, nb_filters, nb_conv, batch_size, nb_epoch)
wpath = 'weights/400ms_new.h5'
mdl.save_weights(wpath)


fname = '/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/C3S_insitu_3_181prj_400ms_248cycles_000.h5'
data = dxchange.read_hdf5(fname,'/exchange/data')
dark = dxchange.read_hdf5(fname,'/exchange/data_dark')
white = dxchange.read_hdf5(fname, '/exchange/data_white')
print data.shape, dark.shape, white.shape
print data.min(), data.max()
data1 = tomopy.normalize(data, white, dark)
print data1.min(), data1.max()
print data1.shape
data1[data1>1] = 0.78
data1[data1<0] = 0
data1 = ds(data1)

predict_patch(data1, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)

# mdl = train_patch(imgx5[0], imgy, patch_size, 2, nb_filters, nb_conv, batch_size, nb_epoch, ipath)
# wpath = 'weights/500ms.h5'
# mdl.save_weights(wpath)
# predict_patch(imgx5, patch_size, 1, nb_filters, nb_conv, batch_size, wpath, spath)



