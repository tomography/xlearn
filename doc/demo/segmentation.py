from xlearn.segmentation import seg_train, seg_predict
import dxchange


batch_size = 4000
nb_epoch = 50
nb_down = 3
nb_gpu = 4

# define the data path
spath = '../../test/seg_data/'

# define the path to save the training weights
wpath = spath + 'weights_seg.h5'

# read the training input and output
imgx = dxchange.read_tiff(spath + 'train_input.tif')
imgy = dxchange.read_tiff(spath + 'train_output.tif')

# train the model
mdl = seg_train(imgx, imgy, batch_size = batch_size, nb_epoch = nb_epoch, nb_down = nb_down, nb_gpu = nb_gpu)

# save the trained weights
mdl.save_weights(wpath)

# read the testing data
img_test = dxchange.read_tiff(spath + 'test_input.tif')

# segmentation for the testing data
seg_predict(img_test, wpath, spath, nb_down = nb_down, nb_gpu = nb_gpu)
