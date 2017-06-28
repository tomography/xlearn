import tomopy
import numpy as np
import dxchange
import matplotlib.pyplot as plt




#
def nor_data(img):
    """
    Normalize the image

    Parameters
    ----------
    img: array
        The images need to be normalized

    Returns
    -------
    img
        Description.
    """
    #     mean_tmp = np.mean(img)
    #     std_tmp = np.std(img)
    #     img = (img - mean_tmp) / std_tmp
    img = img - img.min()
    img = img / img.max()

    return img


def nor_data_ml(img):
    """
    Normalize the image

    Parameters
    ----------
    img: array
        The images need to be normalized

    Returns
    -------
    img
        Description.
    """
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = img + np.abs(img.min())
    img = img + img.mean()
    img = img / img.max()

    return img


def nor_data_pr(img):
    """
    Normalize the image

    Parameters
    ----------
    img: array
        The images need to be normalized

    Returns
    -------
    img
        Description.
    """

    for i in range(len(img)):
        img[i] = nor_data(img[i])
    return img


def nor_data_pr1(img):
    """
    Normalize the image

    Parameters
    ----------
    img: array
        The images need to be normalized

    Returns
    -------
    img
        Description.
    """

    for i in range(len(img)):
        img[:, i, :] = nor_data(img[:, i, :])
    return img


fname = '/home/beams1/YANGX/cnn_prj_enhance/cnn_enhanced_stacks/tf_prd_battery_20170501.tif'
spath = '/home/beams1/YANGX/cnn_prj_enhance/battery_rec/battery1_cnn_20170501'

prj = dxchange.read_tiff(fname)
prj = prj[:-1,500:600,:]
prj = tomopy.prep.normalize.normalize_bg(prj)
# prj = nor_data_ml(prj)
prj = tomopy.minus_log(prj)

prj = tomopy.remove_nan(prj)
print prj.max(), prj.min()
prj = tomopy.remove_neg(prj)
print prj.max(), prj.min()
prj = prj * 500
print prj.max(), prj.min()





theta = tomopy.angles(prj.shape[0])
cen = 464
# dpath = '/home/beams1/YANGX/cnn_prj_enhance/battery_rec/cen'
# tomopy.write_center(prj, theta, dpath, cen_range=[400, 500, 0.5])

# extra_options ={'MinConstraint':0}
# options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
# rec1 = tomopy.recon(prj[-1], theta[:-1], center = cen, algorithm=tomopy.astra, options=options)
rec1 = tomopy.recon(prj[:-1], theta[:-1], center = cen, algorithm = 'gridrec')
dxchange.write_tiff(rec1, spath, dtype = 'float32')


proj_start = 0
proj_end = 1201
ind_tomo = range(proj_start, proj_end)
fname = '/home/beams1/YANGX/cnn_prj_enhance/battery1_ds/prj_00000.tiff'
prj1 = dxchange.read_tiff_stack(fname, ind_tomo, digit = 5)
prj1 = prj1[:-1,500:600,:]
prj1 = tomopy.prep.normalize.normalize_bg(prj1)
# prj = nor_data_ml(prj)
prj1 = tomopy.minus_log(prj1)

prj1 = tomopy.remove_nan(prj1)
print prj.max(), prj.min()
prj1 = tomopy.remove_neg(prj1)
print prj.max(), prj.min()
prj1 = prj1 * 500
print prj.max(), prj.min()

spath = '/home/beams1/YANGX/cnn_prj_enhance/battery_rec/battery1_20170501'
rec2 = tomopy.recon(prj1[:-1], theta[:-1], center = cen, algorithm = 'gridrec')
dxchange.write_tiff(rec2, spath, dtype = 'float32')