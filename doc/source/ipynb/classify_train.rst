Train
-----

Here is an example on how to train a convolutional neural network.

You can download the python scritp :download:`here<../../../doc/demo/transform_train.py>`
or the Jupyter notebook :download:`here<../../../doc/demo/transform_train.ipynb>`

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

    import numpy as np
    from convnet.utils import nor_data
    from convnet.utils import extract_3d
    from convnet.utils import img_window
    from convnet.classify import train

.. code:: python

    np.random.seed(1337)
    dim_img = 128
    patch_size = (dim_img, dim_img)
    batch_size = 50
    nb_classes = 2
    nb_epoch = 12

.. code:: python

    nb_filters = 32
    nb_pool = 2
    nb_conv = 3

    img_y = dxchange.read_tiff('../../test/test_data/training_output.tiff')


to be completed
