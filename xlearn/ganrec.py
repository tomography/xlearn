from __future__ import  absolute_import, division, print_function
import sys
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.framework import ops
from xlearn.utils import nor_data
import matplotlib.pyplot as plt

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
    # conv = tf.map_fn(tf.image.per_image_standardization, conv)
    return conv
#



def dconv2d_norm(x, conv_nb, conv_size, strides, net_init, name, name2):
    conv = tf.compat.v1.layers.conv2d_transpose(x, conv_nb, [conv_size, conv_size], padding='SAME',
                                      strides=strides,
                                      activation=tf.nn.elu, kernel_initializer=net_init, name=name,
                                      reuse=tf.compat.v1.AUTO_REUSE)
    conv = tf.compat.v1.layers.batch_normalization(conv, name=name2, reuse=tf.compat.v1.AUTO_REUSE)
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

        conv7 = conv2d_norm(conv6, 1, 3, (1, 1), net_init, 'conv7a', 'bnconv7a')

        return conv7

def filter_net(inputs, conv_nb, conv_size, dropout, px, reuse = False):
    size_fc = (px//4)** 2
    with tf.compat.v1.variable_scope('generator', reuse=reuse):
        net_init = tf.contrib.layers.variance_scaling_initializer()

        conv1 = conv2d_norm(inputs, conv_nb, conv_size, (1, 1), net_init, 'conv1', 'bnconv1')
        conv1a = conv2d_norm(conv1, conv_nb, conv_size, (2, 2), net_init, 'conv1a', 'bnconv1a')

        conv2 = conv2d_norm(conv1a, conv_nb*2, conv_size, (1, 1), net_init, 'conv2', 'bnconv2')
        conv2a = conv2d_norm(conv2, conv_nb*2, conv_size, (2, 2), net_init, 'conv2a', 'bnconv2a')

        fc1 = tf.compat.v1.layers.flatten(conv2a)
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

        x = tf.compat.v1.layers.conv2d(x, conv_nb, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd1', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.conv2d(x, conv_nb * 2, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd2', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.conv2d(x, conv_nb * 4, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd3', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.conv2d(x, conv_nb * 8, [conv_size, conv_size], strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_initializer=dis_init,
                             name='convd4', reuse=tf.compat.v1.AUTO_REUSE)

        x = tf.compat.v1.layers.flatten(x)
    return x

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
    prj = tfa.image.rotate(prj, ang)
    bp = tf.reduce_mean(prj, 0)
    bp = tf.image.per_image_standardization(bp)
    bp = tf.reshape(bp, [1, bp.shape[0], bp.shape[1], bp.shape[2]])
    return bp

def tomo_radon(rec, ang):
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tfa.image.rotate(img, -ang)
    sino = tf.reduce_mean(img, 1, name=None)
    sino = tf.image.per_image_standardization(sino)
    sino = tf.transpose(sino, [2, 0, 1])
    sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    return sino

def tomo_learn(sinoi, ang, px, reuse, conv_nb, conv_size, dropout, method):
    if method == 'backproj':
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


    recon = tfnor_data(recon)

    sinop = tomo_radon(recon, ang)
    sinop = tfnor_data(sinop)
    return sinop, recon

def cost_mse(ytrue, ypred):
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    return mse

def cost_ssim(ytrue, ypred):

    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tfa.image.ssim(ytrue, ypred, max_val=1))
    return tf.divide(mse, ssim)
    # return 1-tf.reduce_mean(tfa.image.ssim(ytrue, ypred, max_val=1.0))
def cost_ssimms(ytrue, ypred):
    mse = tf.reduce_mean(tf.losses.mean_squared_error(ytrue, ypred))
    ssim = tf.reduce_mean(tfa.image.ssim_multiscale(ytrue, ypred, max_val=1))
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

    pred, recon = tomo_learn(img_input, ang, px, reuse=False, conv_nb=kwargs['conv_nb'],
                             conv_size=kwargs['conv_size'],
                             dropout=kwargs['dropout'],
                             method=kwargs['method']
                             )
    disc_real = discriminator(img_output)
    disc_fake = discriminator(pred, reuse=True)

    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake,
                                                                      labels=tf.ones_like(disc_fake))) \
               + tf.reduce_mean(tf.abs(img_output-pred))*kwargs['cost_rate']

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

            with tf.device('/device:GPU:1'):
                dl, _ = sess.run([disc_loss, train_disc],
                                 feed_dict={img_input: prj, img_output: prj})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: prj})

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


                print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                      ', Discriminator loss = '+ "{:.7f}".format(dl))
        plt.close(fig)
        saver.save(sess, save_wpath)
    return recon

def rec_dcgan(prj, ang, save_wpath, init_wpath = None, **kwargs):
    global g_loss
    ops.reset_default_graph()
    tf.compat.v1.disable_eager_execution()
    cnn_kwargs = ['learning_rate', 'num_steps', 'display_step', 'conv_nb', 'conv_size',
                  'dropout', 'weights_init', 'method','cost_rate', 'gl_tol', 'iter_plot']
    kwargs_defaults = _get_tomolearn_kwargs()
    for kw in cnn_kwargs:
        kwargs.setdefault(kw, kwargs_defaults[kw])
    if init_wpath:
        kwargs['weights_init'] = True

    _, nang, px, _ = prj.shape
    img_input = tf.compat.v1.placeholder(tf.float32, prj.shape)
    img_output = tf.compat.v1.placeholder(tf.float32, prj.shape)

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


    gen_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    disc_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')

    optimizer_gen = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])
    optimizer_disc = tf.compat.v1.train.AdamOptimizer(learning_rate=kwargs['learning_rate'])


    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
 ######################################################################
 # # plots for debug
    if kwargs['iter_plot']:
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
        im3, = axs[1, 1].plot(xdata, g_loss, 'r-')
        tx3 = axs[1, 1].set_title('Generator loss')
        plt.tight_layout()
#########################################################################
    # ani_init()
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
            with tf.device('/device:GPU:1'):
                dl, _ = sess.run([disc_loss, train_disc],
                                 feed_dict={img_input: prj, img_output: prj})
            with tf.device('/device:GPU:2'):
                gl, _ = sess.run([gen_loss, train_gen], feed_dict={img_input: prj, img_output: prj})

            if np.isnan(gl):
                # gl = np.mean(g_loss)
                sess.run(init)
            if kwargs['iter_plot']:
                xdata.append(step)
                g_loss.append(gl)
            # print(g_loss)
                if step % kwargs['display_step'] == 0 or step == 1:
                    pred, recon = sess.run(tomo_learn(prj, ang, px, reuse=True, conv_nb=kwargs['conv_nb'],
                                                      conv_size=kwargs['conv_size'],
                                                      dropout=kwargs['dropout'],
                                                      method=kwargs['method']))
                    ###########################################################
                    sino_plt = np.reshape(pred, (nang, px))
                    sino_plt = np.abs(sino_plt - prj.reshape((nang, px)))
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
                    axs[1, 1].plot(xdata, g_loss, 'r-')
                    plt.pause(0.1)
                    ######################################################################
                    print("Step " + str(step) + ", Generator Loss= " + "{:.7f}".format(gl) +
                          ', Discriminator loss = ' + "{:.7f}".format(dl))
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
        'iter_plot': True
    }

# def ani_init(prj, nang, px):
#     fig, axs = plt.subplots(2, 2, figsize=(16, 8))
#     im0 = axs[0, 0].imshow(prj.reshape(nang, px), cmap='jet')
#     tx0 = axs[0, 0].set_title('Sinogram')
#     fig.colorbar(im0, ax=axs[0, 0])
#     tx1 = axs[1, 0].set_title('Difference of sinogram for iteration 0')
#     im1 = axs[1, 0].imshow(prj.reshape(nang, px), cmap='jet')
#     fig.colorbar(im1, ax=axs[1, 0])
#     im2 = axs[0, 1].imshow(np.zeros((px, px)), cmap='jet')
#     fig.colorbar(im2, ax=axs[0, 1])
#     tx2 = axs[0, 1].set_title('Reconstruction')
#     xdata, g_loss = [], []
#     im3, = axs[1, 1].plot(xdata, g_loss, 'r-')
#     tx3 = axs[1, 1].set_title('Generator loss')
#     plt.tight_layout()
#     return fig, axs, xdata, g_loss

def angles(nang, ang1=0., ang2=180.):
    return np.linspace(ang1 * np.pi / 180., ang2 * np.pi / 180., nang)



