Evaluate
--------

Here is an example on how to use an already trained convolutional neural netwrok
on a new image. The output will be estract features based on the training received.

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

    import numpy as np
    from convnet.utils import nor_data
    from convnet.utils import extract_3d
    from convnet.utils import img_window
    from convnet.classify import model
    import matplotlib.pyplot as plt
    import time
    import glob

.. code:: python

    np.random.seed(1337)

    dim_img = 128
    patch_size = (dim_img, dim_img)
    batch_size = 50
    nb_classes = 2
    nb_epoch = 12
    
to be completed
