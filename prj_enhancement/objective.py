import keras.backend as K
from theano import tensor as T

def loss_DSSIM_theano(y_true, y_pred):
    patches_true = T.nnet.neighbours.images2neibs(y_true, [4, 4])
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4, 4])
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    eps = 1e-9
    std_true = K.sqrt(var_true + eps)
    std_pred = K.sqrt(var_pred + eps)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    return K.mean((1.0 - ssim) / 2.0)

def mSSIM(y_true, y_pred):
    patches_true = T.nnet.neighbours.images2neibs(y_true, [4, 4])
    patches_pred = T.nnet.neighbours.images2neibs(y_pred, [4, 4])
    u_true = K.mean(patches_true, axis=-1)
    u_pred = K.mean(patches_pred, axis=-1)
    var_true = K.var(patches_true, axis=-1)
    var_pred = K.var(patches_pred, axis=-1)
    eps = 1e-9
    std_true = K.sqrt(var_true + eps)
    std_pred = K.sqrt(var_pred + eps)
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    ssim /= denom  # no need for clipping, c1 and c2 make the denom non-zero
    return K.mean(ssim)


def mean_rate(y_true, y_pred):
    return 1-K.abs(K.mean(y_pred-y_true))/K.abs(K.mean(y_true))