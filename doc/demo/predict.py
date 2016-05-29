import os
import re
import dxchange
import time
from convnet.transform import model
from convnet.transform import predict


batch_size = 800
nb_epoch = 40
dim_img = 20
nb_filters = 32
nb_conv = 3
patch_size = (dim_img, dim_img)
patch_step = 4

model = model(dim_img, nb_filters, nb_conv)
model.load_weights('training_weights.h5')

print('Predicting')

folder = '../../convnet/data/'
files = [f for f in sorted(os.listdir(folder)) if re.match(r'.+.tif', f)]

for fname in files:    time_start = time.time()
    sname = fname.split('.')
    time_start = time.time()
    fname_save = folder + sname[0] + '_result'
    img_test = dxchange.read_tiff(folder + fname)
    img_rec = predict(model, img_test, patch_size, patch_step, batch_size, dim_img)
    dxchange.write_tiff(img_rec, fname_save, dtype='float32')
    print(time.time()-time_start)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
       
