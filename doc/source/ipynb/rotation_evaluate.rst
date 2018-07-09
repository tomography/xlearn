Evaluate
--------

Here is an example on how to use an already trained convolutional neural network to
evaluate and select the best image according to the training received. In this example
the network has been trained to select the best rotation axis centered reconstruction.
The test consists of asking the network to select the best centered images coming from a
similar sample collected on a different tomographic beamline.

You can download the python scritp :download:`here<../../../doc/demo/classify_evaluate.py>`
or the Jupyter notebook :download:`here<../../../doc/demo/classify_evaluate.ipynb>`

To run this example please download the test data from the classify_evaluate folder at 
`url <http://tinyurl.com/APS-convnet>`_ 


.. code:: python

    import dxchange
    import numpy as np
    from xlearn.utils import nor_data
    from xlearn.utils import extract_3d
    from xlearn.utils import img_window
    from xlearn.classify import model
    import matplotlib.pyplot as plt
    import time
    import glob


.. parsed-literal::

    Using Theano backend.
    Using gpu device 0: Tesla M2050 (CNMeM is disabled, cuDNN not available)


.. code:: python

    np.random.seed(1337)
    
    dim_img = 128
    patch_size = (dim_img, dim_img)
    batch_size = 50
    nb_classes = 2
    nb_epoch = 12

number of convolutional filters to use

.. code:: python

    nb_filters = 32

size of pooling area for max pooling

.. code:: python

    nb_pool = 2

convolution kernel size

.. code:: python

    nb_conv = 3

Please download the test data from the classify\_evaluate folder at

http://tinyurl.com/APS-xlearn

and put them in the test\_data folder

.. code:: python

    nb_evl = 100

.. code:: python

    fnames = glob.glob('../../test/test_data/*.tiff')
    fnames = np.sort(fnames)

.. code:: python

    mdl = model(dim_img, nb_filters, nb_conv, nb_classes)
    
    mdl.load_weights('classify_training_weights.h5')
    
    Y_score = np.zeros((len(fnames)))


.. code:: python

    for i in range(len(fnames)):
        img = dxchange.read_tiff(fnames[i])
        img = nor_data(img)
        X_evl = np.zeros((nb_evl, dim_img, dim_img))
    
        for j in range(nb_evl):
            X_evl[j] = img_window(img[360:1460, 440:1440], dim_img)
        X_evl = X_evl.reshape(X_evl.shape[0], 1, dim_img, dim_img)
        Y_evl = mdl.predict(X_evl, batch_size=batch_size)
        Y_score[i] = sum(np.dot(Y_evl, [0, 1]))



.. code:: python

    ind_max = np.argmax(Y_score)
    print('The well-centered reconstruction is:', fnames[ind_max])
    plt.plot(Y_score)
    plt.show()


.. parsed-literal::

    ('The well-centered reconstruction is:', '../../test/test_data/1023.00.tiff')



.. image:: rotation_evaluate_files/rotation_evaluate_15_1.png


