import numpy as np
import time
import dxchange
from utils import nor_data, extract_3d, reconstruct_patches
from models import transformer2, transformer3_pooling


def model_choose(ih, iw, nb_conv, size_conv, nb_down, nb_gpu):
    if nb_down == 3:
        mdl = transformer3_pooling(ih, iw, nb_conv, size_conv, nb_gpu)
    else:
        mdl = transformer2(ih, iw, nb_conv, size_conv, nb_gpu)
    return mdl

def seg_train(img_x, img_y, patch_size = 32,
                patch_step = 1, nb_conv = 32, size_conv = 3,
                batch_size =1000, nb_epoch = 20, nb_down = 2, nb_gpu = 1):
    """
    Function description.

    Parameters
    ----------
    img_x: array, 2D or 3D
        Training input of the model. It is the raw image for the segmentation.

    img_y: images, 2D or 3D
        Training output of the model. It is the corresponding segmentation of the training input.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    nb_conv: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    size_conv: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    nb_down: int
          Number of the downsampling for the images in the model.


    Returns
    -------
    mdl
        The trained CNN model for segmenation. The model can be saved for future segmentations.
    """
    # if img_x.ndim == 3:
    #     _, ih, iw = img_x.shape
    # else:
    #     ih, iw = img_x.shape
    patch_shape = (patch_size, patch_size)
    img_x = nor_data(img_x)
    img_y = nor_data(img_y)

    train_x = extract_3d(img_x, patch_shape, patch_step)
    print train_x.shape
    train_y = extract_3d(img_y, patch_shape, patch_step)
    train_x = np.reshape(train_x, (len(train_x), patch_size, patch_size, 1))
    train_y = np.reshape(train_y, (len(train_y), patch_size, patch_size, 1))
    mdl = model_choose(patch_size, patch_size, nb_conv, size_conv, nb_down, nb_gpu)
    print(mdl.summary())
    mdl.fit(train_x, train_y, batch_size=batch_size, epochs=nb_epoch, shuffle=True)
    return mdl

def seg_predict(img, wpath, spath, patch_size = 32, patch_step = 1,
                  nb_conv=32, size_conv=3,
                  batch_size=1000, nb_down=2, nb_gpu = 1):
    """
    the cnn model for image transformation


    Parameters
    ----------
    img : array
        The images need to be segmented.

    wpath: string
        The path where the trained weights of the model can be read.

    spath: string
        The path to save the segmented images.

    patch_size: int
        The size of the small patches extracted from the input images. This size should be big enough to cover the
        features of the segmentation object.

    patch_step: int
         The pixel steps between neighbour patches. Larger steps leads faster speed, but less quality. I recommend 1
         unless you need quick test of the algorithm.

    nb_conv: int
          Number of the covolutional kernals for the first layer. This number doubles after each downsampling layer.

    size_conv: int
          Size of the convolutional kernals.

    batch_size: int
          Batch size for the training. Bigger size leads faster speed. However, it is restricted by the memory size of
          the GPU. If the user got the memory error, please decrease the batch size.

    nb_epoch: int
          Number of the epoches for the training. It can be understand as the number of iterations during the training.
          Please define this number as the actual convergence for different data.

    nb_down: int
          Number of the downsampling for the images in the model.
    Returns
    -------
    y_img
        Description.

      """
    patch_shape = (patch_size, patch_size)
    img = nor_data(img)
    if img.ndim == 2:
        ih, iw = img.shape
        mdl = model_choose(ih, iw, nb_conv, size_conv, nb_down, nb_gpu)
        predict_x = extract_3d(img, patch_shape, patch_step)
        predict_x = np.reshape(predict_x, (len(predict_x), patch_size, patch_size, 1))
        predict_y = mdl.predict(predict_x, batch_size=batch_size)
        predict_y = np.reshape(predict_y, (len(predict_y),patch_size, patch_size))
        predict_y = reconstruct_patches(predict_y, (ih, iw), patch_step)
        fname = spath + 'prd'
        dxchange.write_tiff(predict_y, fname, dtype='float32')
    else:
        pn, ih, iw = img.shape
        mdl = model_choose(ih, iw, nb_conv, size_conv, nb_down, nb_gpu)
        mdl.load_weights(wpath)
        for i in range(pn):
            print('Processing the %s th image' % i)
            tstart = time.time()
            predict_x = img[i]
            predict_x = extract_3d(predict_x, patch_shape, patch_step)
            predict_x = np.reshape(predict_x, (len(predict_x), patch_size, patch_size, 1))
            predict_y = mdl.predict(predict_x, batch_size=batch_size)
            predict_y = np.reshape(predict_y, (len(predict_y), patch_size, patch_size))
            predict_y = reconstruct_patches(predict_y, (ih, iw), patch_step)
            fname = spath + 'prd-' + str(i)
            dxchange.write_tiff(predict_y, fname, dtype='float32')
            print('The prediction runs for %s seconds' % (time.time() - tstart))