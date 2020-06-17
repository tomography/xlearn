from __future__ import  absolute_import, division, print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.keras import Model, layers
import sys
import numpy as np
import dxchange
import time
import matplotlib.pyplot as plt
from PIL import Image


# class Generator(Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = layers.Dense()




def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min())/(img.max()-img.min())
    return img
def nor_prj(img):
    # nang, px = img.shape
    mean_sum = np.mean(np.sum(img, axis=1))
    data_corr = np.zeros_like(img)
    for i in range(len(img)):
        data_corr[i, :] = img[i, :] * mean_sum / np.sum(img[i, :])
    return data_corr

def dense_norm(x, nb_nodes, dropout, net_init, name1, name2):
    fc = tf.layers.dense(x, nb_nodes, activation=tf.nn.tanh, use_bias=True,
                         kernel_initializer=net_init, name=name1,
                          reuse=tf.AUTO_REUSE)
    fc = tf.layers.batch_normalization(fc, name = name2, reuse=tf.AUTO_REUSE)
    fc = tf.layers.dropout(fc, rate = dropout)
    return fc
# def conv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
#     if strides:
#         conv = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size], padding='SAME', strides=strides,
#                                   activation=tf.nn.relu, kernel_initializer=net_init,
#                                   name=name, reuse=tf.AUTO_REUSE)
#     else:
#         conv = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size], padding='SAME', activation=tf.nn.relu,
#                                 kernel_initializer=net_init, name=name, reuse=tf.AUTO_REUSE)
#     conv = tf.layers.batch_normalization(conv, name = name2, reuse=tf.AUTO_REUSE)
#     # conv = tf.map_fn(tf.image.per_image_standardization, conv)
#     return conv
def conv1d(x, conv_nb, conv_size, net_init, name1, name2):
    conv = tf.layers.conv1d(x, conv_nb, conv_size, padding='SAME',
                            activation=tf.nn.softplus, kernel_initializer=net_init,
                            name= name1, reuse=tf.AUTO_REUSE)
    # conv = tf.layers.separable_conv1d(x, conv_nb, conv_size, padding='SAME',
    #                         activation=tf.nn.softplus,
    #                         name= name1, reuse=tf.AUTO_REUSE)
    conv = tf.layers.batch_normalization(conv, name = name2, reuse=tf.AUTO_REUSE)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv

def conv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
    conv = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size], padding='SAME', strides=strides,
                            activation=tf.nn.elu, kernel_initializer=net_init,
                            name=name, reuse=tf.AUTO_REUSE)
    conv = tf.layers.batch_normalization(conv, name = name2, reuse=tf.AUTO_REUSE)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv
#
# def transpose_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
#     if strides:
#         conv = tf.layers.conv2d_transpose(x, conv_nb, [conv_size, conv_size], padding='SAME',
#                                            strides=strides,
#                                            activation=tf.nn.relu, kernel_initializer=net_init, name=name,
#                                            reuse=tf.AUTO_REUSE)
#     else:
#         conv = tf.layers.conv2d_transpose(x, conv_nb, [conv_size, conv_size], padding='SAME',
#                                             activation=tf.nn.relu, kernel_initializer=net_init, name=name,
#                                             reuse=tf.AUTO_REUSE)
#     conv = tf.layers.batch_normalization(conv, name= name2, reuse=tf.AUTO_REUSE)
#     # conv = tf.map_fn(tf.image.per_image_standardization, conv)
#     return conv


def dconv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
    conv = tf.layers.conv2d_transpose(x, conv_nb, [conv_size, conv_size], padding='SAME',
                                      strides=strides,
                                      activation=tf.nn.elu, kernel_initializer=net_init, name=name,
                                      reuse=tf.AUTO_REUSE)
    conv = tf.layers.batch_normalization(conv, name= name2, reuse=tf.AUTO_REUSE)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv


# def conv_net(inputs, conv_nb, conv_size, dropout, px):
#     size_fc = px**2
#     net_init = tf.contrib.layers.variance_scaling_initializer()
#
#     inputs = tf.layers.flatten(inputs)
#     fc1 = dense_norm(inputs, 256, dropout, net_init, 'fc1', 'bn1')
#     fc2 = dense_norm(fc1, 1024, dropout, net_init, 'fc2', 'bn2')
#     # fc2a = dense_norm(fc2, 256, dropout, net_init, 'fc2a', 'bn2a')
#     fc3 = dense_norm(fc2, size_fc, dropout, net_init, 'fcc', 'bn3')
#     conv1 = tf.reshape(fc3, shape=[-1, px, px, 1])
#
#     conv1 = conv2d_norm(conv1, conv_nb, conv_size, net_init, 'conv1')
#     conv1a = conv2d_norm(conv1, conv_nb, conv_size, net_init, 'conv1a', strides = True)
#     conv2 = conv2d_norm(conv1a, conv_nb*2, conv_size+2, net_init, 'conv2')
#     conv2a = conv2d_norm(conv2, conv_nb*2, conv_size+2, net_init, 'conv2a', strides = True)
#     conv3 = conv2d_norm(conv2a, conv_nb*4, conv_size+4, net_init, 'conv3')
#     conv3a = conv2d_norm(conv3, conv_nb*4, conv_size+4, net_init, 'conv3a', strides = True)
#     conv4 = conv2d_norm(conv3a, 1, conv_size+4, net_init, 'conv4')
#
#     fc4 = tf.layers.flatten(conv4)
#     fc4 = dense_norm(fc4, size_fc/64, dropout, net_init, 'fc5', 'bn5')
#     fc4 = dense_norm(fc4, size_fc/256, dropout, net_init, 'fc6', 'bn6')
#     fc4 = dense_norm(fc4, size_fc/64, dropout, net_init, 'fc7', 'bn7')
#     conv5 = tf.reshape(fc4, shape=[-1, px//8, px//8, 1])
#
#     conv5 = transpose_norm(conv5, conv_nb*4, conv_size+4, net_init, 'conv5', strides = True)
#     cc1 = tf.concat([conv5, conv3], axis=3)
#     conv5a = transpose_norm(cc1, conv_nb*4, conv_size+4, net_init, 'conv5a')
#     conv6 = transpose_norm(conv5a, conv_nb*2, conv_size+2, net_init, 'conv6', strides=True)
#     cc2 = tf.concat([conv6, conv2], axis=3)
#     conv6a = transpose_norm(cc2, conv_nb*2, conv_size+2, net_init, 'conv6a')
#     conv7 = transpose_norm(conv6a, conv_nb, conv_size, net_init, 'conv7', strides=True)
#     cc3 = tf.concat([conv7, conv1], axis=3)
#     conv7a = transpose_norm(cc3, conv_nb, conv_size, net_init, 'conv7a')
#     conv8 = transpose_norm(conv7a, conv_nb, conv_size, net_init, 'conv8')
#     conv9 = transpose_norm(conv8, 1, conv_size, net_init, 'conv9')
#     return conv9

# with varialble scaling initializer

# def mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
#     size_fc = px ** 2
#     with tf.variable_scope('generator', reuse=reuse):
#         net_init = tf.contrib.layers.variance_scaling_initializer()
#         inputs = tf.layers.flatten(inputs)
#         fc1 = dense_norm(inputs, 8, dropout, net_init, 'fc1', 'bn1')
#         fc1 = dense_norm(fc1, 8, dropout, net_init, 'fc11', 'bn11')
#         fc1a = dense_norm(fc1, size_fc // 64, dropout, net_init, 'fc1a', 'bn1a')
#         fc2 = dense_norm(inputs, 16, dropout, net_init, 'fc2', 'bn2')
#         fc2 = dense_norm(fc2, 16, dropout, net_init, 'fc22', 'bn22')
#         fc2a = dense_norm(fc2, size_fc // 16, dropout, net_init, 'fc2a', 'bn2a')
#         fc3 = dense_norm(inputs, 32, dropout, net_init, 'fc3', 'bn3')
#         fc3 = dense_norm(fc3, 32, dropout, net_init, 'fc33', 'bn33')
#         fc3a = dense_norm(fc3, size_fc // 4, dropout, net_init, 'fc3a', 'bn3a')
#         fc4 = dense_norm(inputs, 64, dropout, net_init, 'fc4', 'bn4')
#         fc4 = dense_norm(fc4, 64, dropout, net_init, 'fc44', 'bn44')
#         fc4a = dense_norm(fc4, size_fc, dropout, net_init, 'fc4a', 'bn4a')
#
#         conv1 = tf.reshape(fc1a, shape=[-1, px // 8, px // 8, 1])
#
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (8, 8), net_init, 'conv1a', 'bnconv1a')
#
#         conv2 = tf.reshape(fc2a, shape=[-1, px // 4, px // 4, 1])
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (4, 4), net_init, 'conv2a', 'bnconv2a')
#
#         conv3 = tf.reshape(fc3a, shape=[-1, px // 2, px // 2, 1])
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (1, 1), net_init, 'conv3', 'bnconv3')
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (2, 2), net_init, 'conv3a', 'bnconv3a')
#
#         conv4 = tf.reshape(fc4a, shape=[-1, px, px, 1])
#         conv4 = conv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4', 'bnconv4')
#
#         cc = tf.concat([conv1, conv2, conv3, conv4], axis=3)
#         conv5 = conv2d_norm(cc, conv_nb, conv_size, (1, 1), net_init, 'conv5', 'bnconv5')
#
#         conv6 = conv2d_norm(conv5, conv_nb, conv_size, (1, 1), net_init, 'conv6', 'bnconv6')
#
#         conv7 = conv2d_norm(conv6, 1, conv_size, (1, 1), net_init, 'conv7', 'bnconv7')
#
#         return conv7

def mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
    # print inputs.dtype
    size_fc = px ** 2
    with tf.variable_scope('generator', reuse=reuse):
        net_init = tf.contrib.layers.variance_scaling_initializer()
        # inputs = tf.squeeze(inputs, [3])
        # conv1 = conv2d_norm(inputs, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
        # conv1 = conv2d_norm(conv1, conv_nb, conv_size, (1, 1), net_init, 'conv1a', 'bnconv1a')

        # conv1 = conv2d_norm(conv1, 1, conv_size, (1, 1), net_init, 'conv1o', 'bnconv1o')
        # conv1 = conv1d_norm(inputs, 256, 3, 1, net_init, '1con1', '1bnconv1')
        # conv1 = conv1d_norm(conv1, 128, 3, 1, net_init, '1con1a', '1bnconv1a')
        # conv1 = conv1d_norm(conv1, 64, 3, 1, net_init, '1con1b', '1bnconv1b')
        fc = tf.layers.flatten(inputs)
        fc = dense_norm(fc, 128, dropout, net_init, 'fc1', 'bn1')
        fc = dense_norm(fc, 128, dropout,  net_init,'fc1a', 'bn1a')
        fc = dense_norm(fc, 128, dropout, net_init, 'fc1b', 'bn1b')
        # fc = dense_norm(fc, 128, dropout, net_init, 'fc1c', 'bn1c')
        # fc = dense_norm(fc, 128, dropout, net_init, 'fc1d', 'bn1d')
        # fc = dense_norm(fc, 128, dropout, net_init, 'fc1e', 'bn1e')
        # fc = dense_norm(fc, 128, dropout, net_init, 'fc1f', 'bn1f')
        # concat1 = tf.concat([fc1a, fc1b], axis=-1)
        # fc1 = dense_norm(concat1, 128, dropout, net_init, 'fc1c', 'bn1c')
        # fc1 = dense_norm(fc1, 128, dropout, net_init, 'fc1d', 'bn1d')

        fc = dense_norm(fc, size_fc, dropout, net_init, 'fc4a', 'bn4a')



        conv4 = tf.reshape(fc, shape=[-1, px, px, 1])
        conv4 = conv2d_norm(conv4, conv_nb, conv_size+4, (1, 1), net_init, 'conv4', 'bnconv4')
        conv4 = conv2d_norm(conv4, conv_nb, conv_size+2, (1, 1), net_init, 'conv4a', 'bnconv4a')
        conv4 = conv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4b', 'bnconv4b')
        # conv4 = conv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4c', 'bnconv4c')

        #
        conv6 = dconv2d_norm(conv4, conv_nb, conv_size+4, (1, 1), net_init, 'conv5', 'bnconv5')
        conv6 = dconv2d_norm(conv6, conv_nb, conv_size+2, (1, 1), net_init, 'conv6', 'bnconv6')
        conv6 = dconv2d_norm(conv6, conv_nb, conv_size, (1, 1), net_init, 'conv6a', 'bnconv6a')
        # conv6 = dconv2d_norm(conv6, conv_nb, conv_size, (1, 1), net_init, 'conv7', 'bnconv7')
        # conv6 = dconv2d_norm(conv6, conv_nb, conv_size, (1, 1), net_init, 'conv8', 'bnconv8')
        # conv6 = dconv2d_norm(conv6, conv_nb, conv_size, (1, 1), net_init, 'conv9', 'bnconv9')

        conv7 = conv2d_norm(conv6, 1, 3, (1, 1), net_init, 'conv7a', 'bnconv7a')

        return conv7

def filter_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
    # print inputs.dtype
    size_fc = (px//4)** 2
    with tf.variable_scope('generator', reuse=reuse):
        net_init = tf.contrib.layers.variance_scaling_initializer()

        conv1 = conv2d_norm(inputs, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
        conv1a = conv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1a', 'bnconv1a')

        conv2 = conv2d_norm(conv1a, conv_nb*2, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
        conv2a = conv2d_norm(conv2, conv_nb*2, conv_size, (2, 2), net_init, 'conv2a', 'bnconv2a')

        fc1 = tf.layers.flatten(conv2a)
        fc1 = dense_norm(fc1, 128, dropout, net_init, 'fc1', 'bn1')
        fc1a = dense_norm(fc1, 128, dropout, net_init, 'fc1a', 'bn1a')
        # fc1b = dense_norm(fc1a, 128, dropout, net_init, 'fc1b', 'bn1b')
        fc1c = dense_norm(fc1a, size_fc, dropout, net_init, 'fc4a', 'bn4a')
        conv3 = tf.reshape(fc1c, shape=[-1, px//4, px//4, 1])


        conv3 = dconv2d_norm(conv3, conv_nb*2, conv_size, (1, 1), net_init, 'conv3', 'bnconv3')

        conv3a = dconv2d_norm(conv3, conv_nb*2, conv_size, (2, 2), net_init, 'conv3a', 'bnconv3a')
        conv3b = tf.concat([conv2, conv3a], axis=3)
        conv4 = dconv2d_norm(conv3b, conv_nb, conv_size, (1, 1), net_init, 'conv4', 'bnconv4')

        conv4a = dconv2d_norm(conv4, conv_nb, conv_size, (2, 2), net_init, 'conv4a', 'bnconv4a')
        conv4b = tf.concat([conv1, conv4a], axis=3)

        conv5 = dconv2d_norm(conv4b, conv_nb, conv_size, (1, 1), net_init, 'conv5', 'bnconv5')
        conv5 = dconv2d_norm(conv5, conv_nb, conv_size, (1, 1), net_init, 'conv5a', 'bnconv5a')

        conv7 = conv2d_norm(conv5, 1, conv_size, (1, 1), net_init, 'conv7', 'bnconv7')

        return conv7

def conv1d_net(inputs, reuse = False):
    _, px, nang = inputs.shape
    with tf.variable_scope('generator', reuse=reuse):
        net_init = tf.contrib.layers.variance_scaling_initializer()

        conv1 = conv1d(inputs, nang, 3, net_init, 'conv1', 'bnconv1')
        conv2 = conv1d(conv1, nang, 5, net_init, 'conv2', 'bnconv2')


        conv3 = conv1d(conv2, nang, 7, net_init, 'conv3', 'bnconv3')
        conv4 = conv1d(conv3, nang, 5, net_init, 'conv4', 'bnconv4')
        conv5 = conv1d(conv4, px, 3, net_init,  'conv5', 'bnconv5')
        conv6 = conv1d(conv5, px, 3, net_init,  'conv6', 'bnconv6')
        # conv7 = tf.reshape(conv6, shape=[-1, px, px, 1])
        # conv7 = dconv2d_norm(conv7, 32, 3, (1, 1), net_init, 'conv7', 'bnconv7')
        # conv7 = conv2d_norm(conv7, 1, 3, (1, 1), net_init, 'conv7a', 'bnconv7a')

        return conv6




# def mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
#     # print inputs.dtype
#     size_fc = px ** 2
#     with tf.variable_scope('generator', reuse=reuse):
#         net_init = tf.contrib.layers.variance_scaling_initializer()
#         inputs = tf.layers.flatten(inputs)
#         fc1 = dense_norm(inputs, 8, dropout, net_init, 'fc1', 'bn1')
#         fc1 = dense_norm(fc1, 8, dropout,  net_init,'fc11', 'bn11')
#         fc1a = dense_norm(fc1, size_fc // 64, dropout, net_init, 'fc1a', 'bn1a')
#         fc2 = dense_norm(inputs, 16, dropout, net_init, 'fc2', 'bn2')
#         fc2 = dense_norm(fc2, 16, dropout, net_init,  'fc22', 'bn22')
#         fc2a = dense_norm(fc2, size_fc // 16, dropout, net_init, 'fc2a', 'bn2a')
#         fc3 = dense_norm(inputs, 32, dropout, net_init, 'fc3', 'bn3')
#         fc3 = dense_norm(fc3, 32, dropout, net_init, 'fc33', 'bn33')
#         fc3a = dense_norm(fc3, size_fc // 4, dropout, net_init, 'fc3a', 'bn3a')
#         fc4 = dense_norm(inputs, 64, dropout, net_init, 'fc4', 'bn4')
#         fc4 = dense_norm(fc4, 64, dropout, net_init, 'fc44', 'bn44')
#         fc4a = dense_norm(fc4, size_fc, dropout, net_init, 'fc4a', 'bn4a')
#
#         conv1 = tf.reshape(fc1a, shape=[-1, px // 8, px // 8, 1])
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1a', 'bnconv1a')
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1b', 'bnconv1b')
#         conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1c', 'bnconv1c')
#
#         conv2 = tf.reshape(fc2a, shape=[-1, px // 4, px // 4, 1])
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (1, 1), net_init, 'conv2a', 'bnconv2a')
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (2, 2), net_init, 'conv2b', 'bnconv2b')
#         conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (2, 2), net_init, 'conv2c', 'bnconv2c')
#
#
#         conv3 = tf.reshape(fc3a, shape=[-1, px // 2, px // 2, 1])
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (1, 1), net_init, 'conv3', 'bnconv3')
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (1, 1), net_init, 'conv3a', 'bnconv3a')
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (1, 1), net_init, 'conv3b', 'bnconv3b')
#         conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (2, 2), net_init, 'conv3c', 'bnconv3c')
#
#         conv4 = tf.reshape(fc4a, shape=[-1, px, px, 1])
#         conv4 = dconv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4', 'bnconv4')
#         conv4 = dconv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4a', 'bnconv4a')
#         conv4 = dconv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4b', 'bnconv4b')
#         conv4 = dconv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4c', 'bnconv4c')
#
#         cc = tf.concat([tf.image.rot90(conv1), conv2, tf.image.rot90(conv3), conv4], axis=3)
#         conv5 = conv2d_norm(cc, conv_nb, conv_size, (1, 1), net_init, 'conv5', 'bnconv5')
#
#         conv6 = conv2d_norm(conv5, conv_nb, conv_size, (1, 1), net_init, 'conv6', 'bnconv6')
#         # conv6a = conv2d_norm(conv6, conv_nb, conv_size, (1, 1), 'conv6a', 'bnconv6a')
#         # conv6b = conv2d_norm(conv6a, conv_nb, conv_size, (1, 1), 'conv6b', 'bnconv6b')
#
#         conv7 = conv2d_norm(conv6, 1, conv_size, (1, 1), net_init, 'conv7', 'bnconv7')
#
#         return conv7



# #
# def mdnn_net(inputs, conv_nb, conv_size, dropout, px):
#     size_fc = px ** 2
#     net_init = tf.contrib.layers.variance_scaling_initializer()
#     inputs = tf.layers.flatten(inputs)
#     fc1 = dense_norm(inputs, 8, dropout, net_init, 'fc1', 'bn1')
#     fc1 = dense_norm(fc1, 8, dropout, net_init, 'fc11', 'bn11')
#     fc1a = dense_norm(fc1, size_fc // 64, dropout, net_init, 'fc1a', 'bn1a')
#     fc2 = dense_norm(inputs, 16, dropout, net_init, 'fc2', 'bn2')
#     fc2 = dense_norm(fc2, 16, dropout, net_init, 'fc22', 'bn22')
#     fc2a = dense_norm(fc2, size_fc // 16 , dropout, net_init, 'fc2a', 'bn2a')
#     fc3 = dense_norm(inputs, 32, dropout, net_init, 'fc3', 'bn3')
#     fc3 = dense_norm(fc3, 32, dropout, net_init, 'fc33', 'bn33')
#     fc3a = dense_norm(fc3, size_fc // 4, dropout, net_init, 'fc3a', 'bn3a')
#     fc4 = dense_norm(inputs, 64, dropout, net_init, 'fc4', 'bn4')
#     fc4 = dense_norm(fc4, 64, dropout, net_init, 'fc44', 'bn44')
#     fc4a = dense_norm(fc4, size_fc, dropout, net_init, 'fc4a', 'bn4a')
#
#     conv1 = tf.reshape(fc1a, shape=[-1, px // 8, px // 8, 1])
#
#     conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
#     conv1 = dconv2d_norm(conv1, conv_nb, conv_size, (8, 8), net_init, 'conv1a', 'bnconv1a')
#
#     conv2 = tf.reshape(fc2a, shape=[-1, px // 4, px // 4, 1])
#     conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
#     conv2 = dconv2d_norm(conv2, conv_nb, conv_size, (4, 4), net_init, 'conv2a', 'bnconv2a')
#
#
#     conv3 = tf.reshape(fc3a, shape=[-1, px // 2, px // 2, 1])
#     conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (1, 1), net_init, 'conv3', 'bnconv3')
#     conv3 = dconv2d_norm(conv3, conv_nb, conv_size, (2, 2), net_init, 'conv3a', 'bnconv3a')
#
#     conv4 = tf.reshape(fc4a, shape=[-1, px, px, 1])
#     conv4 = conv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4', 'bnconv4')
#
#     cc = tf.concat([conv1, conv2, conv3, conv4], axis=3)
#     conv5 = conv2d_norm(cc, conv_nb, conv_size, (1, 1), net_init, 'conv5', 'bnconv5')
#
#     conv6 = conv2d_norm(conv5, conv_nb, conv_size, (1, 1), net_init, 'conv6', 'bnconv6')
#
#     conv7 = conv2d_norm(conv6, 1, conv_size, (1, 1), net_init, 'conv7', 'bnconv7')
#
#     return conv7

# def mdnn_net(inputs, conv_nb, conv_size, dropout, px):
#     size_fc = px ** 2
#     net_init = tf.contrib.layers.variance_scaling_initializer()
#     inputs = tf.layers.flatten(inputs)
#     fc1 = dense_norm(inputs, 8, dropout, net_init, 'fc1', 'bn1')
#     fc1 = dense_norm(fc1, 8, dropout, net_init, 'fc11', 'bn11')
#     fc1a = dense_norm(fc1, size_fc // 64, dropout, net_init, 'fc1a', 'bn1a')
#     fc2 = dense_norm(inputs, 16, dropout, net_init, 'fc2', 'bn2')
#     fc2 = dense_norm(fc2, 16, dropout, net_init, 'fc22', 'bn22')
#     fc2a = dense_norm(fc2, size_fc // 16 , dropout, net_init, 'fc2a', 'bn2a')
#     fc3 = dense_norm(inputs, 32, dropout, net_init, 'fc3', 'bn3')
#     fc3 = dense_norm(fc3, 32, dropout, net_init, 'fc33', 'bn33')
#     fc3a = dense_norm(fc3, size_fc // 4, dropout, net_init, 'fc3a', 'bn3a')
#     fc4 = dense_norm(inputs, 64, dropout, net_init, 'fc4', 'bn4')
#     fc4 = dense_norm(fc4, 64, dropout, net_init, 'fc44', 'bn44')
#     fc4a = dense_norm(fc4, size_fc, dropout, net_init, 'fc4a', 'bn4a')
#
#     conv1 = tf.reshape(fc1a, shape=[-1, px // 8, px // 8, 1])
#
#     conv1 = tf.layers.conv2d_transpose(conv1, conv_nb, [conv_size, conv_size], padding='SAME',
#                                        activation=tf.nn.relu, kernel_initializer=net_init, name='conv1',
#                                        reuse=tf.AUTO_REUSE)
#     conv1 = tf.layers.conv2d_transpose(conv1, conv_nb, [conv_size, conv_size], padding='SAME',
#                                       strides=(8, 8),
#                                       activation=tf.nn.relu, kernel_initializer=net_init,
#                                       name='conv1a',reuse=tf.AUTO_REUSE)
#
#     conv2 = tf.reshape(fc2a, shape=[-1, px // 4, px // 4, 1])
#     conv2 = tf.layers.conv2d_transpose(conv2, conv_nb, [conv_size, conv_size], padding='SAME',
#                                        activation=tf.nn.relu, kernel_initializer=net_init, name='conv2',
#                                        reuse=tf.AUTO_REUSE)
#     conv2 = tf.layers.conv2d_transpose(conv2, conv_nb, [conv_size, conv_size], padding='SAME',
#                                       strides=(4, 4),
#                                       activation=tf.nn.relu, kernel_initializer=net_init,
#                                       name='conv2a',reuse=tf.AUTO_REUSE)
#     conv3 = tf.reshape(fc3a, shape=[-1, px // 2, px // 2, 1])
#     conv3 = tf.layers.conv2d_transpose(conv3, conv_nb, [conv_size, conv_size], padding='SAME',
#                                        activation=tf.nn.relu, kernel_initializer=net_init, name='conv3',
#                                        reuse=tf.AUTO_REUSE)
#     conv3 = tf.layers.conv2d_transpose(conv3, conv_nb, [conv_size, conv_size], padding='SAME',
#                                       strides=(2, 2),
#                                       activation=tf.nn.relu, kernel_initializer=net_init,
#                                       name='conv3a',reuse=tf.AUTO_REUSE)
#     conv4 = tf.reshape(fc4a, shape=[-1, px, px, 1])
#     conv4 = tf.layers.conv2d(conv4, conv_nb, [conv_size, conv_size], padding='SAME',
#                                       activation=tf.nn.relu, kernel_initializer=net_init, name='conv4',
#                                       reuse=tf.AUTO_REUSE)
#
#     cc = tf.concat([conv1, conv2, conv3, conv4], axis=3)
#     conv5 = tf.layers.conv2d(cc, conv_nb, [conv_size, conv_size], padding='SAME',
#                                       activation=tf.nn.relu, kernel_initializer=net_init, name='conv5',
#                                       reuse=tf.AUTO_REUSE)
#     conv5 = tf.layers.conv2d(conv5, conv_nb//8, [conv_size, conv_size], padding='SAME',
#                                        activation=tf.nn.relu, kernel_initializer=net_init, name='conv6',
#                                        reuse=tf.AUTO_REUSE)
#     conv6 = tf.layers.conv2d(conv5, 1, [conv_size, conv_size], padding='SAME',
#                                        activation=tf.nn.relu, kernel_initializer=net_init, name='conv7',
#                                        reuse=tf.AUTO_REUSE)
#     return conv6


def discriminator(x, reuse = False, conv_nb = 32, conv_size = 3, dropout = 0.25):
    with tf.variable_scope('discriminator', reuse = reuse):
        dis_init = tf.contrib.layers.variance_scaling_initializer()

        x = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd1', reuse=tf.AUTO_REUSE)

        x = tf.layers.conv2d(x, conv_nb * 2, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd2', reuse=tf.AUTO_REUSE)

        x = tf.layers.conv2d(x, conv_nb * 4, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd3', reuse=tf.AUTO_REUSE)

        x = tf.layers.conv2d(x, conv_nb * 8, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd4', reuse=tf.AUTO_REUSE)
        # x = tf.layers.conv2d(x, 1, [conv_size, conv_size], strides=(1, 1), padding='same',
        #                      activation=tf.nn.relu,
        #                      kernel_initializer=dis_init,
        #                      name='convd5')




        #
        # x = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size], padding='same',
        #                     activation=tf.nn.leaky_relu, kernel_initializer=dis_init,
        #                     name='convd1')
        # # x = tf.layers.conv2d(x, conv_nb, [conv_size, conv_size],
        # #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        # #                      name='convd1a')
        # x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, conv_nb*2, [conv_size+2, conv_size+2], padding='same',
        #                     activation=tf.nn.leaky_relu, kernel_initializer=dis_init,
        #                     name='convd2')
        # # x = tf.layers.conv2d(x, conv_nb * 2, [conv_size, conv_size],
        # #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        # #                      name='convd2a')
        # x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, conv_nb*4, [conv_size+4, conv_size+4], padding='same',
        #                     activation=tf.nn.leaky_relu, kernel_initializer=dis_init,
        #                     name='convd3')
        # # x = tf.layers.conv2d(x, conv_nb * 4, [conv_size, conv_size],
        # #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        # #                      name='convd3a')
        # x = tf.layers.average_pooling2d(x, 2, 2)
        # x = tf.layers.conv2d(x, conv_nb * 8, [conv_size + 4, conv_size + 4], padding='same',
        #                      activation=tf.nn.leaky_relu, kernel_initializer=dis_init,
        #                      name='convd4')
        # convd4 = conv2d_norm(convd3, conv_nb * 8, conv_size, (2, 2), dis_init,'convd4', 'bnconvd4')
        # convd5 = conv2d_norm(convd4, 1, conv_size, (1, 1),dis_init, 'convd5', 'bnconvd5')
        x = tf.layers.flatten(x)
        # x = tf.layers.dense(x, 1024, activation=tf.nn.tanh, use_bias=True,
        #                     kernel_initializer=dis_init, name='fc1')
        # x = tf.layers.dense(x, 1024,
        #                     kernel_initializer=dis_init, name='fc1')
        # x = tf.layers.dense(x, 1024, activation=tf.nn.tanh, use_bias=True,
        #                  kernel_initializer=dis_init, name='fc1',
        #                   reuse=tf.AUTO_REUSE)
        # fcd2 = dense_norm(fcd1, 256, dropout, dis_init,'fcd2', 'bnd2')
        # x = tf.layers.dense(x, 2, kernel_initializer=dis_init, name='fc2')

    return x

# def discriminator(x, reuse = False, conv_nb = 32, conv_size = 3, dropout = 0.5):
#     with tf.variable_scope('discriminator', reuse = reuse):
#         dis_init = tf.contrib.layers.variance_scaling_initializer()
#         convd1 = conv2d_norm(x, conv_nb, conv_size, (2, 2), dis_init, 'convd1', 'bnconvd1')
#         convd2 = conv2d_norm(convd1, conv_nb * 2, conv_size, (2, 2), dis_init, 'convd2', 'bnconvd2')
#         convd3 = conv2d_norm(convd2, conv_nb * 4, conv_size, (2, 2), dis_init, 'convd3', 'bnconvd3')
#         convd4 = conv2d_norm(convd3, conv_nb * 8, conv_size, (2, 2), dis_init, 'convd4', 'bnconvd4')
#         convd5 = conv2d_norm(convd4, 1, conv_size, (1, 1), dis_init, 'convd5', 'bnconvd5')
#         fcd1 = tf.layers.flatten(convd5)
#         fcd1 = dense_norm(fcd1, 512, dropout, dis_init, 'fcd1', 'bnd1')
#         fcd2 = dense_norm(fcd1, 256, dropout, dis_init, 'fcd2', 'bnd2')
#         fcd3 = dense_norm(fcd2, 2, dropout, dis_init, 'fcd3', 'bnd3')
#
#     return fcd3



def tfnor_data(img):
    img = (img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))
    return img

def tomo_bp(sinoi, ang):
    prj = tfnor_data(sinoi)
    d_tmp = sinoi.shape
    # print d_tmp
    prj = tf.reshape(prj, [1, d_tmp[1], d_tmp[2], 1])
    prj = tf.tile(prj, [d_tmp[2], 1, 1, 1])
    prj = tf.transpose(prj, [1, 0, 2, 3])
    prj = tf.contrib.image.rotate(prj, ang)
    bp = tf.reduce_mean(prj, 0)
    bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], bp.shape[2]])

    return bp


def tomo_radon(rec, ang):
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tf.contrib.image.rotate(img, -ang)
    sino = tf.reduce_mean(img, 1, name=None)
    sino = tf.image.per_image_standardization(sino)
    sino = tf.transpose(sino, [2, 0, 1])
    sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    return sino


def tomo_learn(sinoi, ang, px, reuse, conv_nb, conv_size, dropout, method):
    if method == 'backproj':
        # print('backproj')
        bp = tomo_bp(sinoi, ang)
        bp = tfnor_data(bp)
        bp = tf.reshape(bp, [bp.shape[0], bp.shape[1], bp.shape[2], 1])
        recon = filter_net(bp, conv_nb, conv_size, dropout, px, reuse=reuse)
    elif method == 'conv1d':
        inputs = tf.reshape(sinoi, [sinoi.shape[0], sinoi.shape[1], sinoi.shape[2]])
        inputs = tf.transpose(inputs, [0, 2, 1])
        recon = conv1d_net(inputs, reuse=reuse)
        recon = tf.reshape(recon, [recon.shape[0], recon.shape[1], recon.shape[2], 1])
    elif method == 'fc':
        inputs = tf.convert_to_tensor(sinoi)
        recon = mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse=reuse)
    else:
        sys.exit('Please provide a correct method. Options: backproj, conv1d, fc')


    # recon = conv_net(inputs, conv_nb, conv_size, dropout, px)

    recon = tfnor_data(recon)

    # print recon.shape
    sinop = tomo_radon(recon, ang)
    # print sinop.shape
    sinop = tfnor_data(sinop)
    # print sinop.shape

    return sinop, recon

def cost_mse(ytrue, ypred):
    # psnr = 1/(10.0 * tf.log(1.0 / (tf.reduce_mean(tf.square(ypred - ytrue)))) / tf.log(10.0))
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))

    return mse

def cost_ssim(ytrue, ypred):
    # psnr = 1/(10.0 * tf.log(1.0 / (tf.reduce_mean(tf.square(tf.subtract(ypred, ytrue))))) / tf.log(10.0))

    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tf.image.ssim(ytrue, ypred, max_val=1))
    return tf.divide(mse, ssim)
    # return 1-tf.reduce_mean(tf.image.ssim(ytrue, ypred, max_val=1.0))
def cost_ssimms(ytrue, ypred):
    # psnr = 1/(10.0 * tf.log(1.0 / (tf.reduce_mean(tf.square(ypred - ytrue)))) / tf.log(10.0))
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tf.image.ssim_multiscale(ytrue, ypred, max_val=1))
    return tf.divide(mse, ssim**0.5)
    # return psnr

def rec_dcgan_back(prj, ang, save_wpath, init_wpath = None, **kwargs):
    tf.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method','cost_rate']
    kwargs_defaults = _get_tomolearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    prj = nor_data(prj)
    img_input = tf.placeholder(tf.float32, prj.shape)
    img_output = tf.placeholder(tf.float32, prj.shape)

    pred, recon = tomo_learn(img_input, ang, px, reuse=False, conv_nb = kwargs['conv_nb'],
                             conv_size = kwargs['conv_size'],
                             dropout = kwargs['dropout'],
                             method = kwargs['method']
                             )
    disc_real = discriminator(img_output)
    disc_fake = discriminator(pred, reuse=True)
    # disc_concat = tf.concat([disc_real, disc_fake], axis=0)

    # stacked_gan = discriminator(pred,  reuse = True)

    # disc_target = tf.placeholder(tf.int32, shape=[None])
    # gen_target = tf.placeholder(tf.int32, shape=[None])

    #
    # gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake,
    #                                                           labels=tf.ones([1], dtype=tf.int32)))
    # disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_real,
    #                                                                 labels=tf.ones([1], dtype=tf.int32)))
    # disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake,
    #                                                                 labels=tf.zeros([1], dtype=tf.int32)))

    # gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_fake,
    #                                                           labels=tf.ones_like(disc_fake)))
    # disc_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_real,
    #                                                                 labels=tf.ones_like(disc_real)))
    # disc_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=disc_fake,
    #                                                                 labels=tf.zeros_like(disc_fake)))
    # sigmoid with L1 loss
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(img_output-pred))*kwargs['cost_rate']
    # sigmoid with L2 loss
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
    #                                                                   labels=tf.ones_like(disc_fake))) \
    #            + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * 1000
    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                            labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                            labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_real+disc_loss_fake

    # disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_concat,
    #                                                                    labels=disc_target))
    # gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan,
    #                                                                      labels=gen_target))

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])


    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)


    # fig, axs =plt.subplots(1, 2)
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        if kwargs['weights_init']:
            if init_wpath == None:
                print('Please provide the file name of initial weights.')
            saver.restore(sess, init_wpath)
        for step in range(1, kwargs['num_steps'] + 1):

            # disc_y = np.concatenate([np.ones([1]), np.zeros([1])], axis=0)
            # gen_y = np.ones([1])
            # feed_dict = {img_input: prj, img_output: prj}
            # _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
            with tf.device('/device:GPU:1'):
                dl, _ = sess.run([disc_loss, train_disc],
                                 feed_dict={img_input: prj, img_output: prj})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: prj})




            # ax = axs[0]
            # ax.plot(gl)
            # if np.isnan(loss):
            #     sess.run(init)

            if step % kwargs['display_step'] == 0 or step == 1:
                pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                  conv_size=kwargs['conv_size'],
                                                  dropout=kwargs['dropout'],
                                                  method=kwargs['method']))
                # if np.isnan(recon.any()) or recon.all()==0:
                #     sess.run(init)
                sino_plt = np.reshape(pred, (nang, px))
                rec_plt = np.reshape(recon, (px, px))
                #
                ax = axs[0]
                ax.imshow(sino_plt, vmax=1, cmap='jet')
                plt.axis('off')
                ax = axs[1]
                ax.imshow(rec_plt, vmax=1, cmap='jet')
                plt.axis('off')
                plt.pause(0.1)


                print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                      ', Discriminator loss = '+ "{:.7f}".format(dl))
        plt.close(fig)
        saver.save(sess, save_wpath)
        # _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb = kwargs['conv_nb'],
        #                      conv_size = kwargs['conv_size'],
        #                      dropout = kwargs['dropout'],
        #                      method = kwargs['method']))
    return recon

def rec_dcgan(prj, ang, save_wpath, init_wpath = None, **kwargs):
    ops.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method','cost_rate', 'gl_tol']
    kwargs_defaults = _get_tomolearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    # prj = nor_data(prj)
    img_input = tf.placeholder(tf.float32, prj.shape)
    img_output = tf.placeholder(tf.float32, prj.shape)

    pred, recon = tomo_learn(img_input, ang, px, reuse=False, conv_nb = kwargs['conv_nb'],
                             conv_size = kwargs['conv_size'],
                             dropout = kwargs['dropout'],
                             method = kwargs['method']
                             )
    disc_real = discriminator(img_output)
    disc_fake = discriminator(pred, reuse=True)

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(img_output - pred)) * kwargs['cost_rate']

    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                            labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                            labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_real+disc_loss_fake


    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    optimizer_disc = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])


    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)



 ######################################################################
 # plots for debug
    fig, axs = plt.subplots(2, 2, figsize=(16, 8))
    im0 = axs[0, 0].imshow(prj.reshape(nang, px), cmap='jet')
    tx0 = axs[0, 0].set_title('Sinogram')
    fig.colorbar(im0, ax=axs[0, 0])
    tx1 = axs[1, 0].set_title('Difference of sinogram for iteration 0')
    im1 = axs[1, 0].imshow(prj.reshape(nang, px), cmap='jet')
    fig.colorbar(im1, ax=axs[1, 0])
    im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='jet')
    fig.colorbar(im2, ax=axs[0, 1])
    tx2 = axs[0, 1].set_title('Reconstruction')
    xdata, g_loss = [], []
    im3, = axs[1, 1].plot(xdata, g_loss)
    tx3 = axs[1, 1].set_title('Generator loss')
    plt.tight_layout()
#########################################################################

    rec_tmp = tf.zeros([1, px, px, 1])

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        if kwargs['weights_init']:
            if init_wpath == None:
                print('Please provide the file name of initial weights.')
            saver.restore(sess, init_wpath)
        for step in range(1, kwargs['num_steps'] + 1):

            # disc_y = np.concatenate([np.ones([1]), np.zeros([1])], axis=0)
            # gen_y = np.ones([1])
            # feed_dict = {img_input: prj, img_output: prj}
            # _, _, gl, dl = sess.run([train_gen, train_disc, gen_loss, disc_loss], feed_dict=feed_dict)
            with tf.device('/device:GPU:1'):
                dl, _ = sess.run([disc_loss, train_disc],
                                 feed_dict={img_input: prj, img_output: prj})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: prj})
            # print(gl, gl.shape, gl.dtype)
            xdata.append(step)
            g_loss.append(gl)
            # print(np.array(g_loss).dtype, np.array(g_loss))


            # ax = axs[0]
            # ax.plot(gl)
            if np.isnan(gl):
                sess.run(init)

            if step % kwargs['display_step'] == 0 or step == 1:
                pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                  conv_size=kwargs['conv_size'],
                                                  dropout=kwargs['dropout'],
                                                  method=kwargs['method']))
                # if (np.isnan(recon.any())) or (recon.all()==0):
                #     sess.run(init)
            ###########################################################
                sino_plt = np.reshape(pred, (nang, px))
                sino_plt = np.abs(sino_plt-prj.reshape((nang, px)))
                rec_plt = np.reshape(recon, (px, px))
                tx1.set_text('Difference of sinogram for iteration {0}'.format(step))
                vmax = np.max(sino_plt)
                vmin = np.min(sino_plt)
                im1.set_data(sino_plt)
                im1.set_clim(vmin, vmax)
                im2.set_data(rec_plt)
                vmax = np.max(rec_plt)
                vmin = np.min(rec_plt)
                im2.set_clim(vmin, vmax)
                im3.set_xdata(xdata)
                im3.set_ydata(g_loss)
                plt.pause(0.1)

            ######################################################################
                print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                      ', Discriminator loss = '+ "{:.7f}".format(dl))
            if gl<kwargs['gl_tol']:
                _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                               conv_size=kwargs['conv_size'],
                                               dropout=kwargs['dropout'],
                                               method=kwargs['method']))
                break
            if step > (kwargs['num_steps'] - 10):
                _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                  conv_size=kwargs['conv_size'],
                                                  dropout=kwargs['dropout'],
                                                  method=kwargs['method']))
                rec_tmp = tf.concat([rec_tmp, recon], axis=0)
                # print(rec_tmp.shape)
        plt.close(fig)
        saver.save(sess, save_wpath)
        # _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb = kwargs['conv_nb'],
        #                      conv_size = kwargs['conv_size'],
        #                      dropout = kwargs['dropout'],
        #                      method = kwargs['method']))
        if rec_tmp.shape[0] >1:
            recon = tf.reduce_mean(rec_tmp, axis=0).eval()


        # print(recon.shape)
    return recon


def rec_cost(prj, ang, save_wpath, log_path, init_wpath = None, **kwargs):
    tf.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method']
    kwargs_defaults = _get_tomolearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    # prj = nor_data(prj)
    X = tf.placeholder('float', prj.shape)
    Y = tf.placeholder('float', prj.shape)
    with tf.name_scope('Model'):
        pred, recon = tomo_learn(X, ang, px, conv_nb=kwargs['conv_nb'],
                                 conv_size=kwargs['conv_size'],
                                 dropout=kwargs['dropout'],
                                 method=kwargs['method'])

    # print prj.shape
    # loss_op = tf.divide(1, tf.reduce_mean(tf.image.ssim(Y, pred, max_val=1)))
    # loss_op = tf.reduce_mean(tf.losses.mean_squared_error(labels=Y, predictions=pred))
    with tf.name_scope('Loss'):
        loss_op = cost_mse(Y, pred)

    with tf.name_scope('Adam'):
        optimizer = tf.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])


    train_op = optimizer.minimize(loss_op)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    #Creat a summary to monitor cost tensor
    tf.summary.scalar("loss", loss_op)
    merged_summary_op = tf.summary.merge_all()
    # x_plot = np.arange(0, kwargs['num_steps'])
    # loss_plot = np.zeros((kwargs['num_steps']))
    fig, axs = plt.subplots(2, 1, figsize=(8, 16))
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        if kwargs['weights_init']:
            if init_wpath == None:
                print('Please provide the file name of initial weights.')
            saver.restore(sess, init_wpath)
        summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())
        for step in range(1, kwargs['num_steps'] + 1):
            # sess.run(train_op, feed_dict={X: prj, Y: prj})
            _, loss, summary = sess.run([train_op, loss_op, merged_summary_op], feed_dict={X: prj, Y: prj})

            summary_writer.add_summary(summary, step)
            # loss_plot[step] = loss
            # plt.plot(x_plot, loss_plot)
            # plt.pause(0.1)
            if np.isnan(loss):
                sess.run(init)
            if step % kwargs['display_step'] == 0 or step == 1:
                pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                  conv_size=kwargs['conv_size'],
                                                  dropout=kwargs['dropout'],
                                                  method=kwargs['method']))


                sino_plt = np.reshape(pred, (nang, px))
                rec_plt = np.reshape(recon, (px, px))
                #
                ax = axs[0]
                ax.imshow(sino_plt, vmax=1, cmap='jet')
                plt.axis('off')
                ax = axs[1]
                ax.imshow(rec_plt, vmax=1, cmap='jet')
                plt.axis('off')
                plt.pause(0.1)
                print("Step " + str(step) + ", Loss= " + "{:.7f}".format(loss))
        plt.close(fig)
        saver.save(sess, save_wpath)
        # pred, recon = sess.run(tomo_learn(prj, ang, px, conv_nb = kwargs['conv_nb'],
        #                      conv_size = kwargs['conv_size'],
        #                      dropout = kwargs['dropout'],
        #                      method = kwargs['method']))
    return  recon

def _get_tomolearn_kwargs():
    return {
        'learning_rate': 5e-3,
        'num_steps': 10000,
        'display_step': 100,
        'conv_nb': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'weights_init': False,
        'method': 'backproj',
        'cost_rate':100,
        'gl_tol': 1.0,
    }

def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)

def center(prj, cen):
    if prj.ndim == 3:
        _, _, px = prj.shape
        cen_diff = px // 2 - cen
        if cen_diff > 0:
            prj = prj[:, :, :-cen_diff * 2]
        if cen_diff < 0:
            prj = prj[:, :, -cen_diff * 2:]
        prj = np.pad(prj, ((0, 0,), (0, 0), (np.abs(cen_diff), np.abs(cen_diff))), 'constant')
    else:
        _, px = prj.shape
        cen_diff = px // 2 - cen
        if cen_diff > 0:
            prj = prj[:, :-cen_diff * 2]
        if cen_diff < 0:
            prj = prj[:, -cen_diff * 2:]
        prj = np.pad(prj, ((0, 0), (np.abs(cen_diff), np.abs(cen_diff))), 'constant')
    return prj

