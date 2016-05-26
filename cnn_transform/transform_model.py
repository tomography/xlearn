from __future__ import print_function
import numpy as np

from keras.models import Sequential
from keras.layers.core import Dense, Reshape, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D



def cnn_model(dim_img, nb_filters, nb_conv):
    """the cnn model for image transformation
    ----------
    dim_img : int
        The input image dimension
    nb_filters : int
        Number of filters
    nb_conv : int
        The convolution weight dimension

    """
    model = Sequential()
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=(1, dim_img, dim_img)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense((dim_img / 4) ** 2))
    model.add(Reshape((1, dim_img / 4, dim_img / 4)))

    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nb_filters * 2, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation('relu'))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(1, 1, 1, border_mode='same'))

    model.compile(loss='mean_squared_error', optimizer='Adam')

    return model


def cnn_prd(img, patch_size, dim_img):
    """the cnn model for image transformation
      ----------
      img : array
          The image need to be calculated
      patch_size : (int, int)
          The patches dimension
      dim_img : int
        The input image dimension

      """
    img = np.float16(nor_data(img))
    img_y, img_x = img.shape
    X_img = extract_patches(img, patch_size, patch_step)
    X_img = np.reshape(X_img, (len(X_img), 1, dim_img, dim_img))
    Y_img = model.predict(X_img, batch_size=batch_size)
    del X_img
    Y_img = np.reshape(Y_img, (len(Y_img), dim_img, dim_img))
    img_rec = reconstruct_patches(Y_img, (img_y, img_x), patch_step)
    return img_rec