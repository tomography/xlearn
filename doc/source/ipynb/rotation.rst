Rotation Center
---------------

An experienced beam line scientist can easily distinguish the well-centered and off-centered reconstructions directly by eyes without any mathematical calculation. This is always the most accurate way to evaluate the results for a final step. However, this approach is not applicable for large data sets because it costs too much effort. Here we use, instead, the classification model of CNN to mimic this process of the human's brain. Thus an automatic routine to compute the tomographic rotation axis is developed.

The rotation axis problem can be considered as an image classification problem, because there are significant different features between a well-center and an off-centered reconstructions. Once the trained CNN can accurately recognize the well-centered reconstruction from the off-centered reconstructions, we can use it as the principle to automatically finding the correct rotation center.

We use here the same classification model described in the CNN introduction, which requires two steps to evaluate the data: train the CNN model, and predict the data with trained model. We developed a special method to prepare the training data sets and to process the prediction.
 
In the procedure of preparing the training data, we reconstruct a slice of tomographic image with different rotation center. A group of reconstruction results are obtained. During the training phase, we select the well-centered reconstruction by eyes, and label the rest as off-centered reconstruction. We extract overlapped patches from the well-centered image and label these patches as 1, and also from the off-centered images and label them as 0.

A patch is a square window (s\ :sub:`p` X s\ :sub:`p`) from the image. The patches are overlapped one by one. The distance of the center between two neighbor patches is the patch steps (*n_s*). The patch number is 

.. math::  N_p = \frac{1}{n_s^2}(h-s_p)\cdot(w-s_p)

for a image with height (*h* ) and width (*w* ). There are two reasons for using small patches instead of the whole image: 

- We can generate enough train data only from one single image. 
- The overlapped patches provide multiple evaluations of the same feature of the image. 

Once we have extracted the patches from the well-centered and off-centered images, we select specific number (*P_{train}*) of patches with their labels (*Y_l* ) from both of these groups. We use the patches as input *X_{train}* and the labels as output to train the CNN model. The trained CNN classification model is now capable to distinguish the well-centered or off-centered patches.

The prediction procedure evaluates the *Y_l* of the patches from the reconstructions of different rotation axis. We first do tomographic reconstruction with different rotation axis. For each reconstructed images, we extract specific number of patches. The size of the patches should be the same as the training data. The number of the patches can be roughly in the hundreds. We use these patches as the input data for trained CNN and evaluate their label. If the value of the label is close to 1, it means the feature of the patch is close to well-centered. If it is 0, it is off-centered. We computer the summation (*S_l* ) of the labels for the patches from one reconstruction. The reconstruction with the maximum *S_l* is the one well-centered as our evaluation model.

.. toctree::

   rotation_train.rst
   rotation_evaluate.rst

.. automodule:: xlearn
   :members:
   :undoc-members:
   :show-inheritance: 
