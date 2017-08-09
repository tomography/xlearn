from skimage.measure import compare_ssim
import dxchange
import numpy as np


def ssimvalue(rec1, rec2):
    ssim_tmp = np.zeros(len(rec1))
    for i in range(len(rec1)):
        img1 = nor_img(rec1[i])
        img2 = nor_img(rec2[i])
        dr = img1.max() - img1.min()
        ssim_tmp[i] =10.0 * np.log(1.0 / (np.mean(np.square(img1 - img2)))) / np.log(10.0)
    return ssim_tmp


def nor_img(img):
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
    img = img[108:620, 108:620]
    mean_tmp = np.mean(img)
    std_tmp = np.std(img)
    img = (img - mean_tmp) / std_tmp
    img = (img - img.min()) / img.max()

    return img

rec_org = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_final/org-1.tiff')
# rec_org = nor_img(rec_org)

rec_n05 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n05.tiff')
# rec_n05 = nor_img(rec_n05)
rec_p05 = dxchange.read_tiff('//home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n05.tiff')
# rec_p05 = nor_img(rec_p05)

rec_n10 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n10.tiff')
# rec_n10 = nor_img(rec_n10)
rec_p10 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n10.tiff')
# rec_p10 = nor_img(rec_p10)

rec_n15 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n15.tiff')
# rec_n15 = nor_img(rec_n15)
rec_p15 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n15.tiff')
# rec_p15 = nor_img(rec_p15)

rec_n20 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n20.tiff')
# rec_n20 = nor_img(rec_n20)
rec_p20 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n20.tiff')
# rec_p20 = nor_img(rec_p20)

rec_n25 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n25.tiff')
# rec_n25 = nor_img(rec_n25)
rec_p25 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n25.tiff')
# rec_p25 = nor_img(rec_p25)

rec_n30 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec_prj2_n/n30.tiff')
# rec_n30 = nor_img(rec_n30)
rec_p30 = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_n30.tiff')
# rec_p30 = nor_img(rec_p30)

ssim_all = np.zeros((12, 512))
ssim_all[0,:] = ssimvalue(rec_org, rec_n05)
ssim_all[1,:] = ssimvalue(rec_org, rec_p05)

ssim_all[2,:] = ssimvalue(rec_org, rec_n10)
ssim_all[3,:] = ssimvalue(rec_org, rec_p10)


ssim_all[4,:] = ssimvalue(rec_org, rec_n15)
ssim_all[5,:] = ssimvalue(rec_org, rec_p15)

ssim_all[6,:] = ssimvalue(rec_org, rec_n20)
ssim_all[7,:] = ssimvalue(rec_org, rec_p20)

ssim_all[8,:] = ssimvalue(rec_org, rec_n25)
ssim_all[9,:] = ssimvalue(rec_org, rec_p25)

ssim_all[10,:] = ssimvalue(rec_org, rec_n30)
ssim_all[11,:] = ssimvalue(rec_org, rec_p30)

np.save('all_psnr_gridrec',ssim_all)