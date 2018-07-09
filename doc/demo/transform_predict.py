import os
import re
import dxchange
import time
from xlearn.transform import model
from xlearn.transform import predict


batch_size = 800
nb_epoch = 40
dim_img = 20
nb_filters = 32
nb_conv = 3
patch_step = 4

patch_size = (dim_img, dim_img)

# define the data path
spath = '../../test/test_data/'

mdl = model(dim_img, nb_filters, nb_conv)
mdl.load_weights(spath + 'transform_training_weights.h5')

print('Predicting')

files = [f for f in sorted(os.listdir(spath)) if re.match(r'.+.tiff', f)]

for fname in files:
    time_start = time.time()
    sname = fname.split('.')
    time_start = time.time()
    fname_save = spath + sname[0] + '_result'
    img_test = dxchange.read_tiff(spath + fname)
    img_rec = predict(mdl, img_test, patch_size, patch_step, batch_size, dim_img)
    dxchange.write_tiff(img_rec, fname_save, dtype='float32')
    print(time.time()-time_start)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
