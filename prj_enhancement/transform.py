import numpy as np
import time
import dxchange
from utils import nor_data, extract_patches, extract_3d, reconstruct_patches
from models import model, model_test, model_filter

def train_full(img_x, img_y, nb_filters, nb_conv, batch_size, nb_epoch):
    """
    Function description.

    Parameters
    ----------
    parameter_01 : type
        Description.

    parameter_02 : type
        Description.

    parameter_03 : type
        Description.

    Returns
    -------
    return_01
        Description.
    """
    pn, iy, ix = img_x.shape


    img_x = nor_data(img_x)
    img_y = nor_data(img_y)


    train_x = np.reshape(img_x, (pn, iy, ix, 1))
    train_y = np.reshape(img_y, (pn, iy, ix, 1))

    mdl = model(iy, ix, nb_filters, nb_conv)
    print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch, shuffle=True)
    return mdl

def train_patch(img_x, img_y, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch):
    """
    Function description.

    Parameters
    ----------
    parameter_01 : type
        Description.

    parameter_02 : type
        Description.

    parameter_03 : type
        Description.

    Returns
    -------
    return_01
        Description.
    """
    # pn, iy, ix = img_x.shape

    patch_shape = (patch_size, patch_size)
    img_x = nor_data(img_x)
    img_y = nor_data(img_y)

    train_x = extract_3d(img_x, patch_shape, patch_step)
    print train_x.shape
    train_y = extract_3d(img_y, patch_shape, patch_step)
    train_x = np.reshape(train_x, (len(train_x), patch_size, patch_size, 1))
    train_y = np.reshape(train_y, (len(train_y), patch_size, patch_size, 1))


    mdl = model_test(patch_size, patch_size, nb_filters, nb_conv)
    # mdl.load_weights(ipath)
    print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch, shuffle=True)
    return mdl


def predict_patch(img, patch_size, patch_step, nb_filters, nb_conv, batch_size, wpath, spath):
    """
    the cnn model for image transformation


    Parameters
    ----------
    img : array
        The image need to be calculated


    Returns
    -------
    y_img
        Description.

      """
    patch_shape = (patch_size, patch_size)
    img = nor_data(img)
    pn, iy, ix = img.shape
    mdl = model_test(patch_size, patch_size, nb_filters, nb_conv)
    mdl.load_weights(wpath)
    for i in range(pn):
        print('Processing the %s th projection' % i)
        tstart = time.time()
        x_img = img[i]
        x_img = extract_3d(x_img, patch_shape, patch_step)
        x_img = np.reshape(x_img, (len(x_img), patch_size, patch_size, 1))
        y_img = mdl.predict(x_img, batch_size=batch_size)
        y_img = np.reshape(y_img, (len(x_img),patch_size, patch_size))
        y_img = reconstruct_patches(y_img, (iy, ix), patch_step)
        fname = spath + 'prj-' + str(i)
        dxchange.write_tiff(y_img, fname, dtype='float32')
        print('The prediction runs for %s seconds' % (time.time() - tstart))

def predict_tf(img, nb_filters, nb_conv, batch_size, wpath, spath):
    """
    the cnn model for image transformation


    Parameters
    ----------
    img : array
        The image need to be calculated


    Returns
    -------
    y_img
        Description.

      """
    img = nor_data(img)
    pn, iy, ix = img.shape
    mdl = model(iy, ix, nb_filters, nb_conv)
    mdl.load_weights(wpath)
    for i in range(pn):
        print('Processing the %s th projection' % i)
        tstart = time.time()
        x_img = img[i]
        x_img = np.reshape(x_img, (1, iy, ix, 1))
        y_img = mdl.predict(x_img, batch_size=batch_size)
        y_img = np.reshape(y_img, (iy, ix))
        fname = spath + 'prj-' + str(i)
        dxchange.write_tiff(y_img, fname, dtype='float32')
        print('The prediction runs for %s seconds' % (time.time() - tstart))

def train_filter(img_x, img_y, patch_size, patch_step, nb_filters, nb_conv, batch_size, nb_epoch):
    """
    Function description.

    Parameters
    ----------
    parameter_01 : type
        Description.

    parameter_02 : type
        Description.

    parameter_03 : type
        Description.

    Returns
    -------
    return_01
        Description.
    """
    pn, iy, ix = img_x.shape

    patch_shape = (patch_size, patch_size)
    img_x = nor_data(img_x)
    img_y = nor_data(img_y)

    train_x = extract_3d(img_x, patch_shape, patch_step)
    train_y = extract_3d(img_y, patch_shape, patch_step)
    train_x = np.reshape(train_x, (len(train_x), patch_size, patch_size, 1))
    train_y = np.reshape(train_y, (len(train_y), patch_size, patch_size, 1))



    mdl = model_filter(patch_size, patch_size, nb_filters, nb_conv)
    print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch, shuffle=True)
    return mdl



def predict_filter(img, patch_size, patch_step, nb_filters, nb_conv, batch_size, wpath, spath):
    """
    the cnn model for image transformation


    Parameters
    ----------
    img : array
        The image need to be calculated


    Returns
    -------
    y_img
        Description.

      """

    img = nor_data(img)
    pn, iy, ix = img.shape
    mdl = model_filter(patch_size, patch_size, nb_filters, nb_conv)
    mdl.load_weights(wpath)
    for i in range(pn):
        print('Processing the %s th projection' % i)
        tstart = time.time()
        x_img = img[i]
        # x_img = extract_3d(x_img, patch_shape, patch_step)
        x_img = np.reshape(x_img, (1, iy, ix, 1))
        y_img = mdl.predict(x_img, batch_size=1)
        y_img = np.reshape(y_img, (iy, ix))

        fname = spath + 'prj-' + str(i)
        dxchange.write_tiff(y_img, fname, dtype='float32')
        print('The prediction runs for %s seconds' % (time.time() - tstart))