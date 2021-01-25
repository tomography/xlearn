from __future__ import  absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.framework import ops
from xlearn.utils import nor_data
from numpy.fft import fftfreq
import matplotlib.pyplot as plt
import dxchange
import time


# class Generator(Model):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.fc1 = layers.Dense()

def nor_data(img):
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    # img = (img - img.min())/(img.max()-img.min())
    return img
def nor_prj(img):
    # nang, px = img.shape
    mean_sum = np.mean(np.sum(img, axis=1))
    data_corr = np.zeros_like(img)
    for i in range(len(img)):
        data_corr[i, :] = img[i, :] * mean_sum / np.sum(img[i, :])
    return data_corr


def dense_norm(x, nb_nodes, dropout, net_init, name1, name2):
    fc = tf.compat.v1.layers.dense(x, nb_nodes, activation=tf.nn.tanh, use_bias=True,
                         kernel_initializer=net_init, name=name1,
                          reuse=tf.compat.v1.AUTO_REUSE)
    fc = tf.compat.v1.layers.batch_normalization(fc, name=name2, reuse=tf.compat.v1.AUTO_REUSE)
    fc = tf.compat.v1.layers.dropout(fc, rate=dropout)
    return fc

def conv1d(x, conv_nb, conv_size, net_init, name1, name2):
    conv = tf.compat.v1.layers.conv1d(x, conv_nb, conv_size, padding='SAME',
                            activation=tf.nn.softplus, kernel_initializer=net_init,
                            name= name1, reuse=tf.compat.v1.AUTO_REUSE)
    # conv = tf.compat.v1.layers.separable_conv1d(x, conv_nb, conv_size, padding='SAME',
    #                         activation=tf.nn.softplus,
    #                         name= name1, reuse=tf.compat.v1.AUTO_REUSE)
    conv = tf.compat.v1.layers.batch_normalization(conv, name =name2, reuse=tf.compat.v1.AUTO_REUSE)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv

def conv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
    conv = tf.compat.v1.layers.conv2d(x, conv_nb, [conv_size, conv_size], padding='SAME', strides=strides,
                            activation=tf.nn.elu, kernel_initializer=net_init,
                            name=name, reuse=tf.compat.v1.AUTO_REUSE)
    conv = tf.compat.v1.layers.batch_normalization(conv, name=name2, reuse=tf.compat.v1.AUTO_REUSE)
    conv = tfnor_data(conv)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv



def dconv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
    conv = tf.compat.v1.layers.conv2d_transpose(x, conv_nb, [conv_size, conv_size], padding='SAME',
                                      strides=strides,
                                      activation=tf.nn.elu, kernel_initializer=net_init, name=name,
                                      reuse=tf.compat.v1.AUTO_REUSE)
    conv = tf.compat.v1.layers.batch_normalization(conv, name=name2, reuse=tf.compat.v1.AUTO_REUSE)
    # conv = tfnor_data(conv)
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv




def mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse=False):
    # print inputs.dtype
    size_fc = px ** 2
    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        # net_init = tf.contrib.layers.variance_scaling_initializer()
        net_init = tf.compat.v1.keras.initializers.VarianceScaling()
        fc = tf.compat.v1.layers.flatten(inputs)
        fc = dense_norm(fc, 128, dropout, net_init, 'fc1', 'bn1')
        fc = dense_norm(fc, 128, dropout,  net_init,'fc1a', 'bn1a')
        fc = dense_norm(fc, 128, dropout, net_init, 'fc1b', 'bn1b')

        fc = dense_norm(fc, size_fc, dropout, net_init, 'fc4a', 'bn4a')

        conv4 = tf.reshape(fc, shape=[-1, px, px, 1])
        conv4 = conv2d_norm(conv4, conv_nb, conv_size+4, (1, 1), net_init, 'conv4', 'bnconv4')
        conv4 = conv2d_norm(conv4, conv_nb, conv_size+2, (1, 1), net_init, 'conv4a', 'bnconv4a')
        conv4 = conv2d_norm(conv4, conv_nb, conv_size, (1, 1), net_init, 'conv4b', 'bnconv4b')
     #
        conv6 = dconv2d_norm(conv4, conv_nb, conv_size+4, (1, 1), net_init, 'conv5', 'bnconv5')
        conv6 = dconv2d_norm(conv6, conv_nb, conv_size+2, (1, 1), net_init, 'conv6', 'bnconv6')
        conv6 = dconv2d_norm(conv6, conv_nb, conv_size, (1, 1), net_init, 'conv6a', 'bnconv6a')

        conv7 = conv2d_norm(conv6, 2, 3, (1, 1), net_init, 'conv7a', 'bnconv7a')

        return conv7

def filter_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
    size_fc = (px//8)** 2
    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        # net_init = tf.contrib.layers.variance_scaling_initializer()
        net_init = tf.compat.v1.keras.initializers.VarianceScaling()

        conv1 = conv2d_norm(inputs, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
        conv1a = conv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1a', 'bnconv1a')

        conv2 = conv2d_norm(conv1a, conv_nb*2, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
        conv2a = conv2d_norm(conv2, conv_nb*2, conv_size, (2, 2), net_init, 'conv2a', 'bnconv2a')

        conv3 = conv2d_norm(conv2a, conv_nb * 2, conv_size, (1, 1), net_init, 'conv3', 'bnconv3')
        conv3a = conv2d_norm(conv3, conv_nb * 2, conv_size, (2, 2), net_init, 'conv3a', 'bnconv3a')

        fc1 = tf.compat.v1.layers.flatten(conv3a)
        fc1 = dense_norm(fc1, 32, dropout, net_init, 'fc1', 'bn1')
        fc1a = dense_norm(fc1, 64, dropout, net_init, 'fc1a', 'bn1a')
        fc1b = dense_norm(fc1a, 128, dropout, net_init, 'fc1b', 'bn1b')
        fc1c = dense_norm(fc1b, size_fc, dropout, net_init, 'fc4a', 'bn4a')
        conv4 = tf.reshape(fc1c, shape=[-1, px//8, px//8, 1])

        conv4 = dconv2d_norm(conv4, conv_nb*2, conv_size, (1, 1), net_init, 'conv4', 'bnconv4')
        # conv4a = tf.concat([conv3a, conv4], axis=3)
        conv4b = dconv2d_norm(conv4, conv_nb*2, conv_size, (2, 2), net_init, 'conv4b', 'bnconv4b')
        conv5 = dconv2d_norm(conv4b, conv_nb*8, conv_size, (1, 1), net_init, 'conv5', 'bnconv5')

        conv5a = tf.concat([conv2a, conv5], axis=3)
        conv5b = dconv2d_norm(conv5a, conv_nb*2, conv_size, (2, 2), net_init, 'conv5b', 'bnconv5b')
        conv6 = dconv2d_norm(conv5b, conv_nb*2, conv_size, (1, 1), net_init, 'conv6', 'bnconv6')

        conv6 = tf.concat([conv1a, conv6], axis=3)
        conv6a = dconv2d_norm(conv6, conv_nb*2, conv_size, (2, 2), net_init, 'conv6a', 'bnconv6a')
        conv6b = dconv2d_norm(conv6a, conv_nb*2, conv_size, (1, 1), net_init, 'conv6b', 'bnconv6b')

        conv7 = tf.concat([conv1, conv6b], axis=3)
        conv7a = dconv2d_norm(conv7, conv_nb, conv_size, (1, 1), net_init, 'conv7a', 'bnconv7a')
        conv7b = dconv2d_norm(conv7a, conv_nb, conv_size, (1, 1), net_init, 'conv7b', 'bnconv7b')
        conv7c = conv2d_norm(conv7b, conv_nb, conv_size, (1, 1), net_init, 'conv7c', 'bnconv7c')
        conv7d = conv2d_norm(conv7c, conv_nb, conv_size, (1, 1), net_init, 'conv7d', 'bnconv7d')

        conv8 = conv2d_norm(conv7d, 2, conv_size, (1, 1), net_init, 'conv8', 'bnconv8')

        return conv8

def conv1d_net(inputs, reuse = False):
    _, px, nang = inputs.shape
    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        net_init = tf.contrib.layers.variance_scaling_initializer()

        conv1 = conv1d(inputs, nang, 3, net_init, 'conv1', 'bnconv1')
        conv2 = conv1d(conv1, nang, 5, net_init, 'conv2', 'bnconv2')

        conv3 = conv1d(conv2, nang, 7, net_init, 'conv3', 'bnconv3')
        conv4 = conv1d(conv3, nang, 5, net_init, 'conv4', 'bnconv4')
        conv5 = conv1d(conv4, px, 3, net_init,  'conv5', 'bnconv5')
        conv6 = conv1d(conv5, px, 3, net_init,  'conv6', 'bnconv6')

        return conv6

def discriminator(x, reuse = False, conv_nb = 32, conv_size = 3, dropout = 0.25):
    with tf.compat.v1.variable_scope('discriminator', reuse = reuse):
        dis_init = tf.compat.v1.keras.initializers.VarianceScaling()
        # dis_init = tf.contrib.layers.variance_scaling_initializer()

        # x = tf.compat.v1.layers.conv2d(x, conv_nb, [conv_size, conv_size], strides=(1, 1), padding='same',
        #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        #                      name='convd1', reuse=tf.compat.v1.AUTO_REUSE)
        x = tf.compat.v1.layers.conv2d(x, conv_nb, [conv_size, conv_size], strides=(2, 2), padding='same',
                                       activation=tf.nn.relu, kernel_initializer=dis_init,
                                       name='convd1a', reuse=tf.compat.v1.AUTO_REUSE)

        # x = tf.compat.v1.layers.conv2d(x, conv_nb * 2, [conv_size, conv_size], strides=(1, 1), padding='same',
        #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        #                      name='convd2', reuse=tf.compat.v1.AUTO_REUSE)
        x = tf.compat.v1.layers.conv2d(x, conv_nb * 2, [conv_size, conv_size], strides=(2, 2), padding='same',
                                       activation=tf.nn.relu, kernel_initializer=dis_init,
                                       name='convd2a', reuse=tf.compat.v1.AUTO_REUSE)
        #
        # x = tf.compat.v1.layers.conv2d(x, conv_nb * 4, [conv_size, conv_size], strides=(1, 1), padding='same',
        #                      activation=tf.nn.relu, kernel_initializer=dis_init,
        #                      name='convd3', reuse=tf.compat.v1.AUTO_REUSE)
        x = tf.compat.v1.layers.conv2d(x, conv_nb * 4, [conv_size, conv_size], strides=(2, 2), padding='same',
                                       activation=tf.nn.relu, kernel_initializer=dis_init,
                                       name='convd3a', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.conv2d(x, conv_nb * 8, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd4', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.flatten(x)
    return x

def tfnor_data(img):
    img = tf.image.per_image_standardization(img)
    img = img/tf.reduce_max(img)
    # img = (img-tf.reduce_min(img))/(tf.reduce_max(img)-tf.reduce_min(img))


    return img
def scale_data(img):
    img = img/1000
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

def phase_fresnel(phase, absorption, h, px):
    paddings = tf.constant([[px // 2, px // 2], [px // 2, px // 2]])
    # padding1 = tf.constant([[px // 2, px // 2], [0, 0]])
    # padding2 = tf.constant([[0, 0], [px // 2, px // 2]])
    pvalue = tf.reduce_mean(phase[:100,:])
    # phase = tf.pad(phase, paddings, 'CONSTANT',constant_values=1)
    phase = tf.pad(phase, paddings, 'SYMMETRIC')
    # phase = tf.pad(phase, paddings, 'REFLECT')
    absorption = tf.pad(absorption, paddings, 'SYMMETRIC')
    # phase = phase
    # absorption = absorption
    abfs = tf.complex(-absorption, phase)
    abfs = tf.exp(abfs)
    ifp = tf.abs(tf.signal.ifft2d(h * tf.signal.fft2d(abfs))) ** 2
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.central_crop(ifp, 0.5)
    ifp = tf.image.per_image_standardization(ifp)
    # ifp = tfnor_data(ifp)
    return ifp

def phase_fraunhofer(phase, absorption):

    wf = tf.complex(absorption, phase)
    # wf = tf.complex(phase, absorption)

    # wf = mask_img(wf)
    # wf = tf.multiply(ampl, tf.exp(phshift))
    # wf = tf.manip.roll(wf, [160, 160], [0, 1])
    ifp = tf.square(tf.abs(tf.signal.fft2d(wf)))
    ifp = tf.roll(ifp, [256, 256], [0, 1])
    ifp = tf.reshape(ifp, [ifp.shape[0], ifp.shape[1], 1])
    ifp = tf.image.per_image_standardization(ifp)
    # ifp = tfnor_data(ifp)
    return ifp


def phase_learn(ifp, h, px, reuse, pure_phase, conv_nb, conv_size, dropout):
    ifp = tf.reshape(ifp, [1, ifp.shape[1], ifp.shape[2], 1])
    # recon = mdnn_net(ifp, conv_nb, conv_size, dropout, px, reuse=reuse)
    # print(recon.shape)
    recon = filter_net(ifp, conv_nb, conv_size, dropout, px, reuse=reuse)
    # recon = tfnor_data(recon)
    # phase = tf.reshape(recon, [px, px])
    # phase = recon[:, :, :, 0]
    phase = tfnor_data(recon[:,:,:,0])-1
    phase = tf.reshape(phase, [px, px])
    # phase = tf.reshape(recon[:,:,:,0], [px, px])
    #
    # phase = tfnor_data(phase)

    # absorption = recon[:, :, :, 1]
    absorption = (1- tfnor_data(recon[:, :, :, 1]))*0.05
    absorption = tf.reshape(absorption, [px, px])
    # absorption = tf.reshape(recon[:,:,:,1], [px, px])
    # absorption = tfnor_data(absorption)

    # absorption = phase
    if pure_phase:
        absorption = tf.zeros_like(phase)
    ifp_prd = phase_fresnel(phase, absorption, h, px)
    # ifp_prd = phase_fraunhofer(phase, absorption)
    ifp_prd = tf.reshape(ifp_prd, [1, ifp.shape[1], ifp.shape[2], 1])
    phase = tf.reshape(phase, [1, phase.shape[0], phase.shape[1], 1])
    absorption = tf.reshape(absorption, [1, absorption.shape[0], absorption.shape[1], 1])
    return ifp_prd, phase, absorption

def tomo_radon(rec, ang):
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tf.contrib.image.rotate(img, -ang)
    sino = tf.reduce_mean(img, 1, name=None)
    # sino = tf.reduce_sum(img, 1, name=None)
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
        # recon = tf.reduce_mean(recon, 3)
        # recon = tf.reshape(recon, [recon.shape[0], recon.shape[1], recon.shape[2], 1])
    else:
        sys.exit('Please provide a correct method. Options: backproj, conv1d, fc')


    # recon = conv_net(inputs, conv_nb, conv_size, dropout, px)
    # print(recon.max(), recon.min())
    recon = tfnor_data(recon)

    # recon = tf.clip_by_value(recon, 0, 1000)
    # recon = tf.image.per_image_standardization(recon)
    # print recon.shape
    sinop = tomo_radon(recon, ang)
    # print sinop.shape
    sinop = tfnor_data(sinop)
    # print sinop.shape

    return sinop, recon

#
# def tomo_train(sinoi, ang, px, reuse, conv_nb, conv_size, dropout, method):
#     if method == 'backproj':
#         # print('backproj')
#         bp = tomo_bp(sinoi, ang)
#         bp = tfnor_data(bp)
#         bp = tf.reshape(bp, [bp.shape[0], bp.shape[1], bp.shape[2], 1])
#         recon = filter_net(bp, conv_nb, conv_size, dropout, px, reuse=reuse)
#     elif method == 'conv1d':
#         inputs = tf.reshape(sinoi, [sinoi.shape[0], sinoi.shape[1], sinoi.shape[2]])
#         inputs = tf.transpose(inputs, [0, 2, 1])
#         recon = conv1d_net(inputs, reuse=reuse)
#         recon = tf.reshape(recon, [recon.shape[0], recon.shape[1], recon.shape[2], 1])
#     elif method == 'fc':
#         inputs = tf.convert_to_tensor(sinoi)
#         recon = mdnn_net(inputs, conv_nb, conv_size, dropout, px, reuse=reuse)
#         # recon = tf.reduce_mean(recon, 3)
#         # recon = tf.reshape(recon, [recon.shape[0], recon.shape[1], recon.shape[2], 1])
#     else:
#         sys.exit('Please provide a correct method. Options: backproj, conv1d, fc')
#
#
#     # recon = conv_net(inputs, conv_nb, conv_size, dropout, px)
#
#     recon = tfnor_data(recon)
#
#     # print recon.shape
#     sinop = tomo_radon(recon, ang)
#     # print sinop.shape
#     sinop = tfnor_data(sinop)
#     # print sinop.shape
#
#     return sinop, recon

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

def rec_dcgan(prj, ang, save_wpath, init_wpath = None, **kwargs):
    tf.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method','cost_rate', 'gl_tol']
    kwargs_defaults = _get_phaselearn_kwargs()
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
                 # + tf.reduce_mean(tf.abs(tf.divide((img_output - pred),
                 #                                    (img_output + 0.95)))) * kwargs['cost_rate']


                #
               # + tf.reduce_mean(tf.abs(tf.divide((img_output - pred), (img_output+1)))) * kwargs['cost_rate']
               # + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * kwargs['cost_rate']
               # + tf.reduce_mean(tf.abs(img_output-pred))*kwargs['cost_rate']
               # + tf.reduce_mean(tf.abs(tf.divide((img_output - pred), (2-img_output)))) * kwargs['cost_rate']

    # sigmoid with L2 loss
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
    #                                                                   labels=tf.ones_like(disc_fake))) \
    #            + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * 1000
    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                            labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                            labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_real+disc_loss_fake
    # clip_op = tf.assign(x, tf.clip(x, 0, np.infty))

    # disc_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = disc_concat,
    #                                                                    labels=disc_target))
    # gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = stacked_gan,
    #                                                                      labels=gen_target))


    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    optimizer_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])


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
    im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='gray')
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

def phase_dcgan(ifp, h, save_wpath, init_wpath = None, **kwargs):
    ops.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    cnn_kwargs = ['pure_phase', 'learning_rate_g', 'learning_rate_d', 'num_steps', 'display_step', 'conv_nb',
                  'conv_size', 'dropout', 'weights_init','cost_rate', 'gl_tol', 'iter_plot']
    kwargs_defaults = _get_phaselearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, px, px, _ = ifp.shape
    img_input = tf.compat.v1.placeholder(tf.float32, ifp.shape)
    img_output = tf.compat.v1.placeholder(tf.float32, ifp.shape)
    ifp = nor_data(ifp)
    pred, phase, absorption = phase_learn(ifp, h, px, reuse=False,
                                          pure_phase= kwargs['pure_phase'],
                                          conv_nb = kwargs['conv_nb'],
                                          conv_size = kwargs['conv_size'],
                                          dropout = kwargs['dropout'])
    disc_real = discriminator(img_output)
    disc_fake = discriminator(pred, reuse=True)

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(img_output - pred)) * kwargs['cost_rate']
               # + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * kwargs['cost_rate']

               # + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * kwargs['cost_rate']



    disc_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real,
                                                                            labels=tf.ones_like(disc_real)))
    disc_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                            labels=tf.zeros_like(disc_fake)))
    disc_loss = disc_loss_real + disc_loss_fake

    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate_g'])
    optimizer_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate_d'])

    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)


 ######################################################################
 # plots for debug
    if kwargs['iter_plot']:
        fig, axs = plt.subplots(2, 3, figsize=(16, 8))
        im0 = axs[0, 0].imshow(ifp.reshape(px, px), cmap='jet')
        tx0 = axs[0, 0].set_title('Intensity')
        fig.colorbar(im0, ax=axs[0, 0])
        tx1 = axs[1, 0].set_title('Absolute error of the intensity for iteration 0')
        im1 = axs[1, 0].imshow(ifp.reshape(px, px), cmap='jet')
        fig.colorbar(im1, ax=axs[1, 0])
        im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='gray')
        fig.colorbar(im2, ax=axs[0, 1])
        tx2 = axs[0, 1].set_title('Phase')
        im3 = axs[0, 2].imshow(np.zeros((px, px)), cmap='jet')
        fig.colorbar(im3, ax=axs[0, 2])
        tx3 = axs[0, 2].set_title('Absorption')
        xdata, g_loss = [], []
        im4, = axs[1, 1].plot(xdata, g_loss, 'r-')
        axs[1, 1].set_yscale('log')
        tx4 = axs[1, 1].set_title('Generator loss')
        plt.tight_layout()
#########################################################################

    rec_tmp = tf.zeros([1, px, px, 1])


    init = tf.compat.v1.global_variables_initializer()
    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
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
                                 feed_dict={img_input: ifp, img_output: ifp})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: ifp, img_output: ifp})
            # print(gl, gl.shape, gl.dtype)
            xdata.append(step)
            g_loss.append(gl)
            # print(np.array(g_loss).dtype, np.array(g_loss))


            # ax = axs[0]
            # ax.plot(gl)
            if np.isnan(gl):
                sess.run(init)

            if step % kwargs['display_step'] == 0 or step == 1:
                pred, phase, absorption = sess.run(phase_learn(ifp, h, px, reuse=False,
                                                   pure_phase= kwargs['pure_phase'],
                                                   conv_nb = kwargs['conv_nb'],
                                                   conv_size = kwargs['conv_size'],
                                                   dropout = kwargs['dropout']))
                # if (np.isnan(recon.any())) or (recon.all()==0):
                #     sess.run(init)
            ###########################################################
                ifp_plt = np.reshape(pred, (px, px))
                ifp_plt = np.abs(ifp_plt - ifp.reshape((px, px)))
                rec_plt = np.reshape(phase, (px, px))
                abs_plt = np.reshape(absorption, (px, px))
                tx1.set_text('Absolute error of the intensity for iteration {0}'.format(step))
                vmax = np.max(ifp_plt)
                vmin = np.min(ifp_plt)
                im1.set_data(ifp_plt)
                im1.set_clim(vmin, vmax)
                im2.set_data(rec_plt)
                vmax = np.max(rec_plt)
                vmin = np.min(rec_plt)
                im2.set_clim(vmin, vmax)
                im3.set_data(abs_plt)
                vmax = np.max(abs_plt)
                vmin = np.min(abs_plt)
                im3.set_clim(vmin, vmax)
                axs[1, 1].plot(xdata, g_loss, 'r-')
                plt.pause(0.1)

            ######################################################################
                print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                      ', Discriminator loss = '+ "{:.7f}".format(dl))
            if gl<kwargs['gl_tol']:
                _, phase, aborption = sess.run(phase_learn(ifp, h, px, reuse=False,
                                                           pure_phase = kwargs['pure_phase'],
                                                           conv_nb = kwargs['conv_nb'],
                                                           conv_size = kwargs['conv_size'],
                                                           dropout = kwargs['dropout']))
                break
            if step > (kwargs['num_steps'] - 10):
                _, phase, absorption = sess.run(phase_learn(ifp, h, px, reuse=False,
                                                            pure_phase = kwargs['pure_phase'],
                                                            conv_nb = kwargs['conv_nb'],
                                                            conv_size = kwargs['conv_size'],
                                                            dropout = kwargs['dropout']))
                rec_tmp = tf.concat([rec_tmp, phase], axis=0)
                # print(rec_tmp.shape)
        plt.close(fig)
        saver.save(sess, save_wpath)
        # _, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb = kwargs['conv_nb'],
        #                      conv_size = kwargs['conv_size'],
        #                      dropout = kwargs['dropout'],
        #                      method = kwargs['method']))
        if rec_tmp.shape[0] >1:
            phase = tf.reduce_mean(rec_tmp, axis=0).eval()


        # print(recon.shape)
    return phase, absorption

def train_dcgan(prj, rec, ang, save_wpath, init_wpath = None, **kwargs):
    tf.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method','cost_rate']
    kwargs_defaults = _get_phaselearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    # prj = nor_data(prj)
    img_input = tf.placeholder(tf.float32, prj.shape)
    img_output = tf.placeholder(tf.float32, rec.shape)

    pred, recon = tomo_learn(img_input, ang, px, reuse=False, conv_nb = kwargs['conv_nb'],
                             conv_size = kwargs['conv_size'],
                             dropout = kwargs['dropout'],
                             method = kwargs['method']
                             )
    disc_real = discriminator(img_output)
    disc_fake = discriminator(recon, reuse=True)
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(tf.divide((img_output-recon), img_output)))*kwargs['cost_rate']
               # + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred))*kwargs['cost_rate']


    # sigmoid with L2 loss
    # gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
    #                                                                   labels=tf.ones_like(disc_fake))) \
    #            + tf.reduce_mean(tf.losses.mean_squared_error(img_output, pred)) * 1000
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
                                 feed_dict={img_input: prj, img_output: rec})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: rec})

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

        # visulazaition update
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


def rec_cost(prj, ang, save_wpath, log_path, init_wpath = None, **kwargs):
    tf.reset_default_graph()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method']
    kwargs_defaults = _get_phaselearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    # prj = nor_data(prj)
    X = tf.placeholder('float', prj.shape)
    Y = tf.placeholder('float', prj.shape)
    with tf.name_scope('Model'):
        pred, recon = tomo_learn(X, ang, px, reuse=False, conv_nb=kwargs['conv_nb'],
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
                pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=False, conv_nb=kwargs['conv_nb'],
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

def _get_phaselearn_kwargs():
    return {
        'pure_phase': True,
        'learning_rate_g': 1e-3,
        'learning_rate_d': 1e-5,
        #######
        # 2e-2 for simulation data
        #######
        'num_steps': 10000,
        'display_step': 100,
        'conv_nb': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'weights_init': False,
        'method': 'backproj',
        'cost_rate':10,
        # For simulation
        # 'gl_tol':0.82,
        ################
        'gl_tol':1e-10,
        'iter_plot': True
    }

def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)

def center(prj, cen):
    _, _, px = prj.shape
    cen_diff = px//2 - cen
    if cen_diff>0:
        prj = prj[:,:,:-cen_diff*2]
    if cen_diff<0:
        prj = prj[:,:,-cen_diff*2:]
    prj = np.pad(prj, ((0, 0,),(0, 0),(np.abs(cen_diff), np.abs(cen_diff))), 'constant')
    return prj

def ffactor(px, energy, z, pv):
    lambda_p = 1.23984122e-09 / energy
    frequ_prefactor = 2 * np.pi * lambda_p * z / pv ** 2
    freq = fftfreq(px)
    xi, eta = np.meshgrid(freq, freq)
    xi = xi.astype('float32')
    eta = eta.astype('float32')
    h = np.exp(- 1j * frequ_prefactor * (xi ** 2 + eta ** 2) / 2)
    return h


