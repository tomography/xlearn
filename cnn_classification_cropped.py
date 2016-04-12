'''Train a simple convnet on the MNIST dataset.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python mnist_cnn.py

Get to 99.25% test accuracy after 12 epochs (there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import matplotlib.pyplot as plt
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.utils.visualize_util import plot

batch_size = 50
nb_classes = 2
nb_epoch = 20

# input image dimensions
img_rows, img_cols = 100, 100
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
texture = np.load('textureCrops.npy')
standard = np.load('standardCrops.npy')


# shuffle them
#np.random.shuffle(texture)
#np.random.shuffle(standard)
#np.random.shuffle(extra)

X_train = np.zeros((2000,100,100))
X_train[0:999] = texture[0:999]
X_train[1000:1999] = standard[0:999]

#X_train[100:199] = 0.7

y_train = np.zeros(2000)
y_train[1000:1999] = 1

X_test = np.zeros((2000,100,100))
X_test[0:999] = texture[2000:2999]
X_test[1000:1999]=standard[1000:1999]
print(X_test.shape)
y_test = np.zeros(2000)
y_test[1000:1999] = 1
print(y_test.shape)
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
#X_train = np.log(X_train)
#X_test = np.log(X_test)
#X_train = (X_train-X_train.max())/(X_train.min()-X_train.max())
#X_test = (X_test-X_test.max())/(X_test.min()-X_test.max())
train_mean = np.mean(X_train)
train_std = np.std(X_train)
X_train = (X_train-train_mean)/train_std
#X_train = abs(X_train/X_train.max())
test_mean = np.mean(X_test)
test_std =np.std(X_test)
X_test = (X_test-test_mean)/test_std
#X_test = abs(X_test/X_test.max())

print(X_test.max(),X_test.min())
#print('adding noise')
#noise_tmp = abs(np.random.normal(0, 0.02,(400,200,200))).astype('float32')
#print('creat noise')
#X_test = np.add(X_test,noise_tmp)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
#X_train[100:199] = 0.7 
#X_test[200:299] = 0.6
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters*2, nb_conv,nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
plot(model, to_file='model.png')
print('Test score:', score[0])
print('Test accuracy:', score[1])

print('Predicting')
start = time.clock()
predicted_output = model.predict(X_test, batch_size=batch_size)
print('The prediction time for 2000 samples is:',time.clock()-start)
np.save('labels',Y_test)
np.save('predicted_output',predicted_output)
print('Predcited class',predicted_output)
i = 1
margin = 5
n = 15
# Visualize the first layer of convolutions on an input image
X = X_test[i:i+1]
img =X_test[0,:,:,:]
img_width = X.shape[2]
img_height = X.shape[3]
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin

stitched_filters=np.zeros((1,width,height))
for i in range(n):
    for j in range(n):
        img =X_test[n,:,:,:]
            stitched_filters[:, (img_width + margin) * i: (img_width + margin)	* i + img_width, (img_height + margin) * j:(img_height + margin) * j + img_height] = img

fb = np.zeros((width,height))
fb = stitched_filters[0]
imsave('conv.png',fb )



# Visualize weights
W = model.layers[0].W.get_value(borrow=True)
W = np.squeeze(W)
print("W shape : ", W.shape[0], W.shape[1:])
n = 6
img_width = W.shape[1]
img_height = W.shape[2]
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin

stitched_filters=np.zeros((1,width,height))
for i in range(n):
    for j in range(n):
        index=i*n+j
            if index < W.shape[0]:
                img =W[j]
                stitched_filters[:, (img_width + margin) * i: (img_width + margin)  * i + img_width, (img_height + margin) * j:(img_height + margin) * j + img_height] = img

fb = np.zeros((width,height))
fb = stitched_filters[0]
imsave('weight.png',fb )





# Visualize convolution result (after activation)
convout1_f = theano.function([model.get_input(train=False)], convout1.get_output(train=False))
W = convout1_f(X)
W = np.squeeze(W)
print("C1 shape : ", W.shape)

n=6
img_width = W.shape[1]
img_height = W.shape[2]
width = n * img_width + (n - 1) * margin
height = n * img_height + (n - 1) * margin

stitched_filters=np.zeros((1,width,height))
for i in range(n):
    for j in range(n):
        index=i*n+j
            if index < W.shape[0]:
                #			print("index is", index)
                img =W[j]
                    stitched_filters[:, (img_width + margin) * i: (img_width + margin)  * i + img_width, (img_height + margin) * j:(img_height + margin) * j + img_height] = img
ff = np.zeros((width,height))
ff = stitched_filters[0]
plt.imshow(ff)
plt.show()
imsave('conf1.png',ff )

print('Ploting Results')
Y_predicted = np.zeros(len(predicted_output))
for i in range(len(predicted_output)):
    if np.round(predicted_output[i,0]) ==1:
       Y_predicted[i] = 0
    else:
       Y_predicted[i] = 1
       
xxx = range(len(Y_test))
plt.subplot(2, 1, 1)
plt.scatter(xxx,Y_test)
plt.title('Expected')
plt.ylim((-0.2, 1.2))
plt.subplot(2, 1, 2)
plt.scatter(xxx,Y_predicted)
plt.title('Predicted')
plt.ylim((-0.2, 1.2))
plt.show()
