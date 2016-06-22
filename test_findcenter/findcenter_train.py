import dxchange
import numpy as np
from utils import nor_data
from utils import extract_3d
from utils import img_window
from classify import train

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





fname = '/home/oxygen/YANGX/Globus/center/Blakely_SLS_1038_1058_1/1038.tiff'
ind_uncenter1 = range(1038, 1047)
ind_uncenter2 = range(1049, 1057)
uncenter1 = dxchange.read_tiff_stack(fname, ind=ind_uncenter1, digit=4)
uncenter2 = dxchange.read_tiff_stack(fname, ind=ind_uncenter2, digit=4)
uncenter = np.concatenate((uncenter1, uncenter2), axis=0)
uncenter = nor_data(uncenter)
print uncenter.shape
uncenter = img_window(uncenter[:, 360:1460, 440:1440], 200)
print uncenter.shape
uncenter_patches = extract_3d(uncenter, patch_size, 1)
np.random.shuffle(uncenter_patches)
print uncenter_patches.shape
# print uncenter_patches.shape
center_img = dxchange.read_tiff('/home/oxygen/YANGX/Globus/center/Blakely_SLS_1038_1058_1/1048.tiff')
center_img = nor_data(center_img)
print center_img.shape
center_img = img_window(center_img[360:1460, 440:1440], 400)
center_patches = extract_3d(center_img, patch_size, 1)
np.random.shuffle(center_patches)
print center_patches.shape
# plt.imshow(center_img, cmap='gray', interpolation= None)
# plt.show()

x_train = np.concatenate((uncenter_patches[0:50000], center_patches[0:50000]), axis=0)
x_test = np.concatenate((uncenter_patches[50000:60000], center_patches[50000:60000]), axis=0)
x_train = x_train.reshape(x_train.shape[0], 1, dim_img, dim_img)
x_test = x_test.reshape(x_test.shape[0], 1, dim_img, dim_img)
y_train = np.zeros(100000)
y_train[50000:99999] = 1
y_test = np.zeros(20000)
y_test[10000:19999] = 1

train(x_train, y_train, x_test, y_test, dim_img, nb_filters, nb_conv, batch_size, nb_epoch, nb_classes)
#
# y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
# print x_train.shape, y_train.shape, x_test.shape, Y_test.shape
#
# model = Sequential()
#
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
#                         border_mode='valid',
#                         input_shape=(1, img_rows, img_cols)))
# model.add(Activation('relu'))
# model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
#
# model.add(Convolution2D(nb_filters*2, nb_conv,nb_conv))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=["accuracy"])
#
# model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
#           show_accuracy=True, verbose=1, validation_data=(x_test, Y_test))
# model.save_weights('weight_center.h5')
# score = model.evaluate(x_test, Y_test, show_accuracy=True, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
#
# print('Predicting')
# predicted_output = model.predict(x_test, batch_size=batch_size)
# np.save('labels', Y_test)
# np.save('predicted_output', predicted_output)
