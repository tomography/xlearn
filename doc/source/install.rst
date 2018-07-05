=======
Install
=======

This section covers the basics of how to download and install 
`xlearn <https://github.com/tomography/xlearn>`_.

.. contents:: Contents:
   :local:


Installing from source
======================

1. Install anaconda (https://conda.io/docs/user-guide/install/index.html)

2. Install tensorflow (https://www.tensorflow.org/install/install_linux). Please install the tensorflow-gpu version with
pip. Before the tensorflow-gpu installation, make sure that the cuda drivers and cudnn are correctly installed to your OS.

3. Install the Xlearn toolbox:
Clone the `xlearn <https://github.com/tomography/xlearn>`_
from `GitHub <https://github.com>`_ repository::

    git clone https://github.com/tomography/xlearn.git xlearn

then::

    cd xlearn
    python setup.py install
    
