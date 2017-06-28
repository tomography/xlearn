from keras.models import Model
from keras import backend as K
from keras.layers.core import Dense, Reshape, Flatten, Dropout, Lambda
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose

# from padding import mirror_padding, padding_shape, mirror_padding2, padding_shape2
# from objective_tf import DSSIMObjective

def psnr(y_true, y_pred):
    return 1/(10.0 * K.log(1.0 / (K.mean(K.square(y_pred - y_true)))) / K.log(10.0))



def model(iy, ix, nb_filters, nb_conv):

    inputs = Input((iy, ix, 1))

    conv1 = Conv2D(nb_filters, ((nb_conv, nb_conv)), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(conv1)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(1, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    #
    fc1 = Flatten()(conv4)
    fc1 = Dense(ix * iy / 128, activation='relu')(fc1)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Dense(ix * iy / 128, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Dense(ix * iy / 64, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Reshape((iy / 8, ix / 8, 1))(fc1)

    fc2 = Conv2DTranspose(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(fc1)

    fc2 = Conv2DTranspose(nb_filters * 8, (nb_conv, nb_conv), activation='relu', padding='same')(fc2)

    up1 = merge([UpSampling2D(size=(2, 2))(fc2), conv3], mode='concat', concat_axis=3)


    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv6)

    up2 = merge([UpSampling2D(size=(2, 2))(conv6), conv2], mode='concat',concat_axis=3)

    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv7)

    up2 = merge([UpSampling2D(size=(2, 2))(conv7), conv1], mode='concat',concat_axis=3)

    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(up2)
    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(conv8)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)



    mdl = Model(inputs=inputs, outputs=conv8)

    mdl.compile(loss= 'mse', optimizer='Adam', metrics=['accuracy'])
    return mdl



def model_test(iy, ix, nb_filters, nb_conv):

    inputs = Input((iy, ix, 1))

    conv1 = Conv2D(nb_filters, ((nb_conv, nb_conv)), activation='relu', padding='same')(inputs)
    conv1a = Conv2D(nb_filters, (nb_conv, nb_conv),
                   strides=(2,2),activation='relu', padding='same')(conv1)



    conv2 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv1a)
    conv2a = Conv2D(nb_filters * 2, (nb_conv, nb_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv2)


    conv3 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv2a)
    conv3a = Conv2D(nb_filters * 2, (nb_conv, nb_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv3)


    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv3a)
    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(1, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    #
    fc1 = Flatten()(conv4)
    fc1 = Dense(ix * iy / 128, activation='relu')(fc1)
    fc1 = Dropout(0.2)(fc1)
    fc1 = Dense(ix * iy / 128, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Dense(ix * iy / 64, activation='relu')(fc1)
    fc1 = Dropout(0.25)(fc1)
    fc1 = Reshape((iy / 8, ix / 8, 1))(fc1)

    fc2 = Conv2DTranspose(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(fc1)

    fc2 = Conv2DTranspose(nb_filters * 8, (nb_conv, nb_conv),
                          strides=(2, 2), activation='relu', padding='same')(fc2)

    up1 = merge([fc2, conv3], mode='concat', concat_axis=3)


    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv6)

    up2 = merge([conv6, conv2], mode='concat',concat_axis=3)

    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv7)

    up3 = merge([conv7, conv1], mode='concat',concat_axis=3)

    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(up3)
    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(conv8)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)



    mdl = Model(inputs=inputs, outputs=conv8)

    mdl.compile(loss= psnr, optimizer='Adam', metrics=['mse'])
    return mdl




def model_filter(iy, ix, nb_filters, nb_conv):

    inputs = Input((iy, ix, 1))

    conv1 = Conv2D(nb_filters, ((nb_conv, nb_conv)), activation='relu', padding='same')(inputs)
    conv1a = Conv2D(nb_filters, (nb_conv, nb_conv),
                   strides=(2,2),activation='relu', padding='same')(conv1)



    conv2 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv1a)
    conv2a = Conv2D(nb_filters * 2, (nb_conv, nb_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv2)


    conv3 = Conv2D(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(conv2a)
    conv3a = Conv2D(nb_filters * 2, (nb_conv, nb_conv),
                   strides=(2, 2), activation='relu', padding='same')(conv3)


    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv3a)
    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    conv4 = Conv2D(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)
    #

    conv5 = Conv2DTranspose(nb_filters * 4, (nb_conv, nb_conv), activation='relu', padding='same')(conv4)

    conv5 = Conv2DTranspose(nb_filters * 8, (nb_conv, nb_conv),
                          strides=(2, 2), activation='relu', padding='same')(conv5)

    up1 = merge([conv5, conv3], mode='concat', concat_axis=3)


    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up1)
    conv6 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv6)

    up2 = merge([conv6, conv2], mode='concat',concat_axis=3)

    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv), activation='relu', padding='same')(up2)
    conv7 = Conv2DTranspose(nb_filters * 2, (nb_conv, nb_conv),
                            strides=(2, 2), activation='relu', padding='same')(conv7)

    up3 = merge([conv7, conv1], mode='concat',concat_axis=3)

    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(up3)
    conv8 = Conv2DTranspose(nb_filters, (nb_conv, nb_conv), activation='relu', padding='same')(conv8)

    conv8 = Conv2DTranspose(1, (3, 3), activation='relu', padding='same')(conv8)



    mdl = Model(inputs=inputs, outputs=conv8)

    mdl.compile(loss= 'mse', optimizer='Adam', metrics=['accuracy'])
    return mdl

