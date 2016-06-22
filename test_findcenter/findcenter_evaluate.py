import dxchange
import numpy as np
from utils import nor_data
from utils import extract_3d
from utils import img_window
from classify import model
import matplotlib.pyplot as plt
import time
import glob
np.random.seed(1337)

dim_img = 128
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
nb_evl = 100

start_time = time.time()
fnames = glob.glob('/home/oxygen/YANGX/Globus/center/test_05/Hornby_APS_1000_1058_1/*.tiff')
fnames = np.sort(fnames)

mdl = model(dim_img, nb_filters, nb_conv, nb_classes)

mdl.load_weights('weight_center.h5')
print('The model loading time is %s seconds'%(time.time()-start_time))
start_time = time.time()
Y_score = np.zeros((len(fnames)))

for i in range(len(fnames)):
    img = dxchange.read_tiff(fnames[i])
    img = nor_data(img)
    X_evl = np.zeros((nb_evl, dim_img, dim_img))

    for j in range(nb_evl):
        X_evl[j] = img_window(img[360:1460, 440:1440], dim_img)
    X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)
    Y_evl = mdl.predict(X_evl, batch_size=batch_size)
    Y_score[i] = sum(np.dot(Y_evl, [0, 1]))
    #print('The evaluate score is:', Y_score[i])
    #Y_score = sum(np.round(Y_score))/len(Y_score)


ind_max = np.argmax(Y_score)
print('The well-centered reconstruction is:', fnames[ind_max])
print('The prediction runs for %s seconds'%(time.time()-start_time))
plt.plot(Y_score)
plt.show()


