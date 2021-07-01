# Salienecy Detection in Medical Images #
A collection of projects focused on medical image analysis. They make use of TensorFlow (particularly its Keras API) to create Neural Networks. They rely on openenly available data so that they can be easily reused.

## Overview ##
The following is a brief summary of the projects. 
* __Histopathologic Cancer Detection__: provides a cancer detection system to detect metastatic breast cancer from small patches of pathology whole slide images (WSI) from the Patch Camelyon dataset. The solution applies transfer learning to a ResNet50 model updated with a binary classification head. Gradient-weighted Class Activtion Mapping (GradCAM) is used to identify the regions of the images that have a high probability of containing malignant cells.   

## Histopathologic Cancer Detection ##
Lymph nodes are typically the first place breast cancer spreads and is therefore detection of lymph node metastases is an important part of identifying the extent of the cancer. Manual inspection of histopathologic slides is a laborious process and therefore prone to human error and this makes it a good candidate for automation through machinbe learning.

We build a binary image classifier to identify the presence of metastases in 96 x 96 pixel patches from digital histopathology images where the metastases can exist in a small number of cells. The histopathological images are microscopic images of lymph nodes that have been stained with hematoxylin and eosin (H&E) which produces blue, red and violet colors. Hematoxylin binds to nucleic acids producing blue coloration and eosin binds to amino-acid chains producing pink coloration. This differentiation means that cell nuclei are typically stained blue but cytoplasm and extracellular material is stained pink. We use the Patch Camelyon (PCam) dataset which contains 220K training patches obtained from 400 H&E stained whole slide images collected by Radboud University Medical Center (Nijmegen, Netherlands) and University Medical Center Utrecht (Utrecht, Netherlands).

### Preprocessing and image augmentation ###
The PCam dataset is not large and we therefore have to find ways to avoid overfitting and these may include data augmentaion, regularization and employing less complex architectures. The following is an outline of the image augmentions that were applied at train time using the ImageDataGenerator supplied by Keras.
* Random Roation: 0 to 90 degrees
* Random Zoom: 0% to 20%
* Random Horizontal Flip
* Random Vertical Flip
* Random Translation: 0% to 10%

TODO: example of image augmentaions

### Model selection ###
Image augmentation helps to combat overfitting but we can help further by choosing a smaller model architecture and use transfer learning to update pre-trained model weights to our dataset. Models that are small and lend themselves to transfer learning include DenseNet169, MobileNetV2 and ResNet50. We will choose ResNet50 which although not the smallest netowork in the list it combats overfitting by using residual layers.

TODO: training and evaluation sections

### Gradient-weighted Class Activation Mapping (Grad-CAM) ###

