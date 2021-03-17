# Convolutional Neural Networks in TensorFlow #
A collection of notebooks for projects focused on image classification problems. They make opinionated use of TensorFlow (particularly its Keras API) to create Convolutional Neural Networks (convnets) and rely on openenly available data so that they can be easily run.

## Overview ##
The following is a brief summary of the projects. 
* __Histopathologic Cancer Detection__: provides a cancer detection system to detect metastatic breast cancer from small patches of pathology whole slide images (WSI) from the Patch Camelyon dataset. The solution applies transfer learning to a ResNet50 model updated with a binary classification head. Gradient-weighted Class Activtion Mapping (GradCAM) is used to identify the regions of the images that have a high probability of containing malignant cells.   

## Histopathologic Cancer Detection ##

