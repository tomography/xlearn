import dxchange
import tomopy
import numpy as np

def prj_pro(prj):
    prj = prj + np.abs(prj.min())
    prj = prj + prj.mean()
    print prj.max(), prj.min()
    prj = prj / prj.max()
    print prj.max(), prj.min()
    prj = tomopy.minus_log(prj)
    prj = tomopy.remove_nan(prj)
    print prj.max(), prj.min()
    prj = tomopy.remove_neg(prj)
    print prj.max(), prj.min()

    prj = prj * 500
    print prj.max(), prj.min()
    return prj


prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n05.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n05', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n10.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n10', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n15.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n15', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n20.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n20', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n25.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n25', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_prd_tf/tf_n30.tif')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/tf_sirt_n30', dtype = 'float32')










prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_05.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n05', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_10.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n10', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_15.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n15', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_20.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n20', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_25.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n25', dtype = 'float32')

prj = dxchange.read_tiff('/home/beams/YANGX/cnn_prj_enhance/sim/prj2_noise/n_30.tiff')
prj = prj_pro(prj)
theta = tomopy.angles(prj.shape[0])
cen = 363.5
extra_options ={'MinConstraint':0}
options = {'proj_type':'cuda', 'method':'SIRT_CUDA', 'num_iter':200, 'extra_options':extra_options}
rec1 = tomopy.recon(prj, theta, center = cen, algorithm=tomopy.astra, options=options)
print rec1.max(), rec1.min()
dxchange.write_tiff(rec1, '/home/beams/YANGX/cnn_prj_enhance/sim/rec2_prd_tf/sirt_n30', dtype = 'float32')
