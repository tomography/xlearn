Predict
-------

Here is an example on how to use an already trained convolutional neural network
to automatically segment a series of raw images.

You can download the python scritp :download:`here<../../../doc/demo/transform_predict.py>`
or the Jupyter notebook :download:`here<../../../doc/demo/transform_predict.ipynb>`

.. code:: python

    %pylab inline


.. parsed-literal::

    Populating the interactive namespace from numpy and matplotlib


.. code:: python

    import dxchange

Image data I/O in ConvNet is supported by 
`DXchange <http://dxchange.readthedocs.io>`__.

.. code:: python

    import matplotlib.pyplot as plt

matplotlib provide plotting of the result in this notebook.

`Install <http://convnet.readthedocs.io/en/latest/install.html>`__ ConvNet
then:

.. code:: python

    from convnet.transform import model
    from convnet.transform import predict

.. code:: python

    batch_size = 800
    nb_epoch = 40
    dim_img = 20
    nb_filters = 32
    nb_conv = 3
    patch_step = 4
    
    patch_size = (dim_img, dim_img)

.. code:: python

    mdl = model(dim_img, nb_filters, nb_conv)
    mdl.load_weights('training_weights.h5')

.. code:: python

    fname = '../../test/test_data/predict_test.tiff'
    img_test = dxchange.read_tiff(fname)
    plt.imshow(img_test, cmap='Greys_r')
    plt.show()



.. image:: transform_predict_files/predict_4_0.png


.. code:: python

    fname_save = '../../test/test_data/predict_test_result'

.. code:: python

    img_rec = predict(mdl, img_test, patch_size, patch_step, batch_size, dim_img)

.. code:: python

    dxchange.write_tiff(img_rec, fname_save, dtype='float32')

.. code:: python

    plt.imshow(img_rec, cmap='Greys_r')
    plt.show()



.. image:: transform_predict_files/predict_8_0.png


