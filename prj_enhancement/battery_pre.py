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

data = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/battery_201703/H01C16_0.1_5_252.h5',
                          '/exchange/data')
dark = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/battery_201703/H01C16_0.1_5_252.h5',
                          '/exchange/data_dark')
white = dxchange.read_hdf5('/home/beams1/YANGX/cnn_prj_enhance/battery_201703/H01C16_0.1_5_252.h5',
                          '/exchange/data_white')

print data.min(), data.max()

data = tomopy.normalize(data, white, dark)

data[data>10] = 0.5
data[data<-10] = 0
print data.min(), data.max()
data = ds(data)
data = data[:, 0:950, 130:1090]
print data.min(), data.max()

dxchange.write_tiff_stack(data, '/home/beams1/YANGX/cnn_prj_enhance/battery1_ds/prj', dtype = 'float32')