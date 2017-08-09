import tomopy
import numpy as np
import dxchange
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim
from skimage.transform import downscale_local_mean

def ds(img):
    il, ih, iw = img.shape
    img_tmp = np.zeros((il, ih / 2, iw / 2))
    for i in range(len(img)):
        img_tmp[i] = downscale_local_mean(img[i], (2, 2))
    return img_tmp

data = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/training_insitu_1_6650eV_5X_5_prj_100msx300_0_1681.h5',
                          '/exchange/data')
dark = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/training_insitu_1_6650eV_5X_5_prj_100msx300_0_1681.h5',
                          '/exchange/data_dark')
white = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/training_insitu_1_6650eV_5X_5_prj_100msx300_0_1681.h5',
                          '/exchange/data_white')

print data.shape, dark.shape, white.shape
print data.min(), data.max()
data = tomopy.normalize(data[:300], white, dark)
data[data>10] = 0.5
data[data<-10] = 0
print data.min(), data.max()
print data.shape

data = ds(data)

dxchange.write_tiff(data, '/home/beams1/YANGX/cnn_prj_enhance/exposure_tests/ds_data', dtype='float32')