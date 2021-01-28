# Convnets in Tensorflow #
A collection of notebooks for projects focused on image classification problems. They make opinionated use of TensorFlow (particularly its Keras API) to create Convolutional Neural Networks (convnets) and rely on openenly available data so that they can be easily run.

## Overview ##
The following is a brief summary of the projects available. 
* __Binary image classification__: predict images as belonging to one of two classes using a labeled dataset. A baseline model is compiled that resembles the original convnets of Yann LeCun (http://yann.lecun.com/exdb/publis/pdf/lecun-99.pdf). To combat overfitting use of image augmentation is provide using the Keras preprocessing API and dropout layers are included in the model.

## Future work ##
The initial baseline model for binary image classification will be improved upon with transfer learning and use a dataset of histopatholog scans of lymph node sections (PatchCamelyon) will be used to provide a more applied use case (breast cancer detection). 
