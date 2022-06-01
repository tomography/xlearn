"""
Module containing GANtomo, GANrec
"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from xlearn.models import make_generator, make_discriminator, make_filter
from xlearn.utils import RECONmonitor


__authors__ = "Xiaogang Yang"
__copyright__ = "Copyright (c) 2022, Brookhaven National Laboratory & DESY"
__version__ = "0.3.0"
__docformat__ = "restructuredtext en"
__all__ = ['GANtomo',
           'GANrec']

# @tf.function
def discriminator_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_output,
                                                                       labels=tf.ones_like(real_output)))
    fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                       labels=tf.zeros_like(fake_output)))
    total_loss = real_loss + fake_loss
    return total_loss


def l1_loss(img1, img2):
    return tf.reduce_mean(tf.abs(img1 - img2))
def l2_loss(img1, img2):
    return tf.square(tf.reduce_mean(tf.abs(img1-img2)))



# @tf.function
def generator_loss(fake_output, img_output, pred, l1_ratio):
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                      labels=tf.ones_like(fake_output))) \
               + l1_loss(img_output, pred) * l1_ratio
    return gen_loss


# @tf.function
def filer_loss(fake_output, img_output, img_filter):
    f_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_output,
                                                                    labels=tf.ones_like(fake_output))) + \
              l1_loss(img_output, img_filter) *10
              # l1_loss(img_output, img_filter) * 10
    return f_loss


def tfnor_data(img):
    img = (img - tf.reduce_min(img)) / (tf.reduce_max(img) - tf.reduce_min(img))
    return img


def tfnor_phase(img):
    img = tf.image.per_image_standardization(img)
    img = img / tf.reduce_max(img)
    return img


def avg_results(recon, loss):
    sort_index = np.argsort(loss)
    recon_tmp = recon[sort_index[:10], :, :, :]
    return np.mean(recon_tmp, axis=0)


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


@tf.function
def tomo_radon(rec, ang):
    nang = ang.shape[0]
    img = tf.transpose(rec, [3, 1, 2, 0])
    img = tf.tile(img, [nang, 1, 1, 1])
    img = tfa.image.rotate(img, -ang, interpolation='bilinear')
    sino = tf.reduce_mean(img, 1, name=None)
    sino = tf.image.per_image_standardization(sino)
    sino = tf.transpose(sino, [2, 0, 1])
    sino = tf.reshape(sino, [sino.shape[0], sino.shape[1], sino.shape[2], 1])
    return sino


class GANrec:
    def __init__(self, prj_input, angle, **kwargs):
        rec_args = _get_GANtomo_kwargs()
        rec_args.update(**kwargs)
        super(GANtomo, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = rec_args['iter_num']
        self.conv_num = rec_args['conv_num']
        self.conv_size = rec_args['conv_size']
        self.dropout = rec_args['dropout']
        self.l1_ratio = rec_args['l1_ratio']
        self.g_learning_rate = rec_args['g_learning_rate']
        self.d_learning_rate = rec_args['d_learning_rate']
        self.save_wpath = rec_args['save_wpath']
        self.init_wpath = rec_args['init_wpath']
        self.init_model = rec_args['init_model']
        self.recon_monitor = rec_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.prj_input.shape[0],
                                  self.prj_input.shape[1])
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    # def make_chechpoints(self):
    #     checkpoint_dir = '/data/ganrec/training_checkpoints'
    #     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    #     checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
    #                                      discriminator_optimizer=self.discriminator_optimizer,
    #                                      generator=self.generator,
    #                                      discriminator=self.discriminator)

    @tf.function
    def recon_step(self, img_input):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(img_input)
            recon = tfnor_data(recon)
            img_forward = forward_model(recon)
            img_forward = tfnor_data(img_forward)
            real_output = self.discriminator(img_input, training=True)
            fake_output = self.discriminator(img_forward, training=True)
            g_loss = generator_loss(fake_output, img_input, img_forward, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_rec': img_forward,
                'g_loss': g_loss,
                'd_loss': d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            f_loss = filer_loss(filter_output, prj, prj_filter)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_filter = filter_tape.gradient(f_loss,
                                                   self.filter.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.filter_optimizer.apply_gradients(zip(gradients_of_filter,
                                                  self.filter.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_filter': prj_filter,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step

            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.recon_step(prj, ang)
            step_result = self.recon_step(prj, ang)
            # step_result = self.recon_step_filter(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            # recon[epoch, :, :, :], prj_rec, gen_loss[epoch], d_loss = self.train_step_filter(prj, ang)
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result['prj_rec'], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].numpy()))
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')
        return recon[epoch]
        # return avg_results(recon, gen_loss)


class GANtomo:
    """
    Tomographic reconstruction with GANrec algorithm.

    Parameters
    ----------
    prj_input: array, 2D
        Input sinogram for reconstruction.

    angle: array, 2D
        Tomographic scanning angles for reconstruction.

    iter_num: int
       Iteration number of reconstruction.

    conv_num: int
          Number of the covolutional kernals for the first layer. Default value is 32.

    conv_size: int
          Size of the convolutional kernals. Default value is 3.

    dropout: int
         dropout for fully connected layers. Default value is 0.25

    l1: int
          Ratio of l1 loss adding to gan loss. Default value is 100.

    g_learning_rate: float
          Learning rate of the generator. Default value is 1e-3

    d_learning_rate: float
          Learning rate of the discriminator. Default value is 1e-5

    save_wpath: strings
          The file nane and path of saving the trained weights after reconstruction.

    init_wpath: strings
          The file nane and path of initial weights for reconstruction.
    recon_monitor: bool
          Turn on or off the convergence plot for the reconstruction.


    Returns
    -------
    2D array
        The tomographic reconstruction result.
    """
    def __init__(self, prj_input, angle, **kwargs):
        tomo_args = _get_GANtomo_kwargs()
        tomo_args.update(**kwargs)
        super(GANtomo, self).__init__()
        self.prj_input = prj_input
        self.angle = angle
        self.iter_num = tomo_args['iter_num']
        self.conv_num = tomo_args['conv_num']
        self.conv_size = tomo_args['conv_size']
        self.dropout = tomo_args['dropout']
        self.l1_ratio = tomo_args['l1_ratio']
        self.g_learning_rate = tomo_args['g_learning_rate']
        self.d_learning_rate = tomo_args['d_learning_rate']
        self.save_wpath = tomo_args['save_wpath']
        self.init_wpath = tomo_args['init_wpath']
        self.init_model = tomo_args['init_model']
        self.recon_monitor = tomo_args['recon_monitor']
        self.filter = None
        self.generator = None
        self.discriminator = None
        self.filter_optimizer = None
        self.generator_optimizer = None
        self.discriminator_optimizer = None

    def make_model(self):
        self.filter = make_filter(self.prj_input.shape[0],
                                  self.prj_input.shape[1])
        self.generator = make_generator(self.prj_input.shape[0],
                                        self.prj_input.shape[1],
                                        self.conv_num,
                                        self.conv_size,
                                        self.dropout,
                                        1)
        self.discriminator = make_discriminator(self.prj_input.shape[0],
                                                self.prj_input.shape[1])
        self.filter_optimizer = tf.keras.optimizers.Adam(5e-5)
        self.generator_optimizer = tf.keras.optimizers.Adam(self.g_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(self.d_learning_rate)
        self.generator.compile()
        self.discriminator.compile()

    # def make_chechpoints(self):
    #     checkpoint_dir = '/data/ganrec/training_checkpoints'
    #     checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    #     checkpoint = tf.train.Checkpoint(generator_optimizer=self.generator_optimizer,
    #                                      discriminator_optimizer=self.discriminator_optimizer,
    #                                      generator=self.generator,
    #                                      discriminator=self.discriminator)

    @tf.function
    def recon_step(self, prj, ang):
        # noise = tf.random.normal([1, 181, 366, 1])
        # noise = tf.cast(noise, dtype=tf.float32)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            recon = self.generator(prj)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            g_loss = generator_loss(fake_output, prj, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    def recon_step_filter(self, prj, ang):
        with tf.GradientTape() as filter_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # tf.print(tf.reduce_min(sino), tf.reduce_max(sino))
            prj_filter = self.filter(prj)
            prj_filter = tfnor_data(prj_filter)
            recon = self.generator(prj_filter)
            recon = tfnor_data(recon)
            prj_rec = tomo_radon(recon, ang)
            prj_rec = tfnor_data(prj_rec)
            real_output = self.discriminator(prj, training=True)
            filter_output = self.discriminator(prj_filter, training=True)
            fake_output = self.discriminator(prj_rec, training=True)
            f_loss = filer_loss(filter_output, prj, prj_filter)
            g_loss = generator_loss(fake_output, prj_filter, prj_rec, self.l1_ratio)
            d_loss = discriminator_loss(real_output, fake_output)
        gradients_of_filter = filter_tape.gradient(f_loss,
                                                   self.filter.trainable_variables)
        gradients_of_generator = gen_tape.gradient(g_loss,
                                                   self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss,
                                                        self.discriminator.trainable_variables)
        self.filter_optimizer.apply_gradients(zip(gradients_of_filter,
                                                  self.filter.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(gradients_of_generator,
                                                     self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,
                                                         self.discriminator.trainable_variables))
        return {'recon': recon,
                'prj_filter': prj_filter,
                'prj_rec': prj_rec,
                'g_loss': g_loss,
                'd_loss': d_loss}

    @property
    def recon(self):
        nang, px = self.prj_input.shape
        prj = np.reshape(self.prj_input, (1, nang, px, 1))
        prj = tf.cast(prj, dtype=tf.float32)
        prj = tfnor_data(prj)
        ang = tf.cast(self.angle, dtype=tf.float32)
        self.make_model()
        if self.init_wpath:
            self.generator.load_weights(self.init_wpath+'generator.h5')
            print('generator is initilized')
            self.discriminator.load_weights(self.init_wpath+'discriminator.h5')
        recon = np.zeros((self.iter_num, px, px, 1))
        gen_loss = np.zeros((self.iter_num))

        ###########################################################################
        # Reconstruction process monitor
        if self.recon_monitor:
            plot_x, plot_loss = [], []
            recon_monitor = RECONmonitor('tomo')
            recon_monitor.initial_plot(self.prj_input)
        ###########################################################################
        for epoch in range(self.iter_num):

            ###########################################################################
            ## Call the rconstruction step
            step_result = self.recon_step(prj, ang)
            recon[epoch, :, :, :] = step_result['recon']
            gen_loss[epoch] = step_result['g_loss']
            ###########################################################################

            plot_x.append(epoch)
            plot_loss = gen_loss[:epoch + 1]
            if (epoch + 1) % 100 == 0:
                # checkpoint.save(file_prefix=checkpoint_prefix)
                if recon_monitor:
                    prj_rec = np.reshape(step_result['prj_rec'], (nang, px))
                    prj_diff = np.abs(prj_rec - self.prj_input.reshape((nang, px)))
                    rec_plt = np.reshape(recon[epoch], (px, px))
                    recon_monitor.update_plot(epoch, prj_diff, rec_plt, plot_x, plot_loss)
                print('Iteration {}: G_loss is {} and D_loss is {}'.format(epoch + 1,
                                                                           gen_loss[epoch],
                                                                           step_result['d_loss'].numpy()))
            # plt.close()
        if self.save_wpath != None:
            self.generator.save(self.save_wpath+'generator.h5')
            self.discriminator.save(self.save_wpath+'discriminator.h5')
        return recon[epoch]
        # return avg_results(recon, gen_loss)


def _get_GANtomo_kwargs():
    return{
        'iter_num': 1000,
        'conv_num': 32,
        'conv_size': 3,
        'dropout': 0.25,
        'l1_ratio': 100,
        'g_learning_rate': 1e-3,
        'd_learning_rate': 1e-5,
        'save_wpath': None,
        'init_wpath': None,
        'init_model': False,
        'recon_monitor': True,
    }

