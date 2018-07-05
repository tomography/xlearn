from xlearn.segmentation import seg_train, seg_predict
import dxchange


batch_size = 4000
nb_epoch = 50
nb_down = 3
nb_gpu = 4

# define the path to save the training weights
wpath = '../../test/seg_data/weights_seg.h5'
# define the path to save the results
spath = '../../test/seg_data/'
# read the training input and output
imgx = dxchange.read_tiff('../../test/seg_data/train_input.tif')
imgy = dxchange.read_tiff('../../test/seg_data/train_output.tif')
# train the model
mdl = seg_train(imgx, imgy, batch_size = batch_size, nb_epoch = nb_epoch, nb_down = nb_down, nb_gpu = nb_gpu)
# save the trained weights
mdl.save_weights(wpath)
# read the testing data
img_test = dxchange.read_tiff('../../test/seg_data/test_input.tif')
# segmentation for the testing data
seg_predict(img_test, wpath, spath, nb_down = nb_down, nb_gpu = nb_gpu)
