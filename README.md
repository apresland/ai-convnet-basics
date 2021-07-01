# Saliency Detection in Medical Images #

<table style="width:100%">
  <tr>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124128454-1ac7c900-da7d-11eb-8ef4-24509bf7133f.png">
           <img src="https://user-images.githubusercontent.com/5468707/124128454-1ac7c900-da7d-11eb-8ef4-24509bf7133f.png"
           width="200" height="200"></a>
           <br>non-tumor, p = 0.03
      </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124128457-1c918c80-da7d-11eb-942a-8f5a7583ad6d.png">
           <img src="https://user-images.githubusercontent.com/5468707/124128457-1c918c80-da7d-11eb-942a-8f5a7583ad6d.png"
            width="200" height="200"></a>
           <br>non-tumor map
        </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124128465-1e5b5000-da7d-11eb-934f-e24750afca84.png">
           <img src="https://user-images.githubusercontent.com/5468707/124128465-1e5b5000-da7d-11eb-934f-e24750afca84.png"
            width="200" height="200"></a>
           <br>tumor, p = 0.99
        </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124128476-20bdaa00-da7d-11eb-8f61-238c78ed5255.png">
           <img src="https://user-images.githubusercontent.com/5468707/124128476-20bdaa00-da7d-11eb-8f61-238c78ed5255.png"
            width="200" height="200"></a>
           <br>tumor map
        </p>
    </th>
  </tr>
</table>

__Figure 1__: Histopathologic samples of (a) normal tissue and (c) cancerous tissue correctly classified with a convolutional neural network and the same samples overlayed GradCAM activation map (AM) for (b) normal tissue and (d) cancerous tissue.

## Overview ##
The following is a brief summary of the project. 
* __Histopathologic Cancer Detection__: provides a cancer detection system to detect metastatic breast cancer from small patches of pathology whole slide images (WSI) from the Patch Camelyon dataset. The solution applies transfer learning to a ResNet50 model updated with a binary classification head. Gradient-weighted Class Activtion Mapping (GradCAM) is used to identify the regions of the images that have a high probability of containing malignant cells.   
* __Gradient-weighted Class Activation Mapping__: makes Convolutional Neural Network (CNN)-based models more transparent by visualizing the regions of input that are "salient" for predictions from these models. It uses the class-specific gradient information passed into the final convolutional layer of a CNN to produce a localization map of the important regions in the image.

## Histopathologic Cancer Detection ##
<img src="https://user-images.githubusercontent.com/5468707/124146697-4f904c00-da8e-11eb-8f34-0d12d2982c00.jpeg" width="275" height="180"/>

Lymph nodes are typically the first place breast cancer spreads and is therefore detection of lymph node metastases is an important part of identifying the extent of the cancer. Manual inspection of histopathologic slides is a laborious process and therefore prone to human error and this makes it a good candidate for automation through machinbe learning.

We build a binary image classifier to identify the presence of metastases in 96 x 96 pixel patches from digital histopathology images where the metastases can exist in a small number of cells. The histopathological images are microscopic images of lymph nodes that have been stained with hematoxylin and eosin (H&E) which produces blue, red and violet colors. Hematoxylin binds to nucleic acids producing blue coloration and eosin binds to amino-acid chains producing pink coloration. This differentiation means that cell nuclei are typically stained blue but cytoplasm and extracellular material is stained pink. We use the Patch Camelyon (PCam) dataset which contains 220K training patches obtained from 400 H&E stained whole slide images collected by Radboud University Medical Center (Nijmegen, Netherlands) and University Medical Center Utrecht (Utrecht, Netherlands).

## Gradient-weighted Class Activation Mapping (Grad-CAM) ##
[Grad-CAM](https://arxiv.org/abs/1610.02391) is a technique for producing "visual explanations" for decisions from CNN-based models, making them more transparent it uses the gradients flowing into the final convolutional layer to produce a localization map highlighting important regions in the image for predicting the class. It can be applied as a computational saliency technique in medical imaging for of abnormality detection and computer-aided diagnosis.

## Preprocessing and image augmentation ##

<table style="width:100%">
  <tr>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124138485-f96bda80-da86-11eb-8fe0-2676711af92e.png">
           <img src="https://user-images.githubusercontent.com/5468707/124138485-f96bda80-da86-11eb-8fe0-2676711af92e.png"
           width="200" height="200"></a>
           <br>augmentation #1
      </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124138487-fb359e00-da86-11eb-8523-8a34458b829f.png">
           <img src="https://user-images.githubusercontent.com/5468707/124138487-fb359e00-da86-11eb-8523-8a34458b829f.png"
            width="200" height="200"></a>
           <br>augmentation #2
        </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124138490-fcff6180-da86-11eb-9935-d538dff8a761.png">
           <img src="https://user-images.githubusercontent.com/5468707/124138490-fcff6180-da86-11eb-9935-d538dff8a761.png"
            width="200" height="200"></a>
           <br>augmentation #3
        </p>
    </th>
    <th><p>
           <a href="https://user-images.githubusercontent.com/5468707/124138500-fec92500-da86-11eb-90de-cc705638557c.png">
           <img src="https://user-images.githubusercontent.com/5468707/124138500-fec92500-da86-11eb-90de-cc705638557c.png"
            width="200" height="200"></a>
           <br>augmentation #4
        </p>
    </th>
  </tr>
</table>

__Figure 2__: Augmentations of the same original image as output by the ImageDataGenerator in Keras.

The PCam dataset is not large and we therefore have to find ways to avoid overfitting and these may include data augmentaion, regularization and employing less complex architectures. The following is an outline of the image augmentions that were applied at train time using the ImageDataGenerator supplied by Keras.
* Random Roation: 0 to 90 degrees
* Random Zoom: 0% to 20%
* Random Horizontal Flip
* Random Vertical Flip
* Random Translation: 0% to 10%

## Model selection ##
Image augmentation helps to combat overfitting but we can help further by choosing a smaller model architecture and use transfer learning to update pre-trained model weights to our dataset. Models that are small and lend themselves to our task include DenseNet169, MobileNetV2 and ResNet50. We will also build a custom model that we will call CancerNet.

## Model Training ##
The training curves shown in Figure 3 demonstrate that by using augmentation and choosing small networks we have succefully compated overfitting while still achieving an accuracy close to 95% for all models. 

<img src="https://user-images.githubusercontent.com/5468707/124140749-0be71380-da89-11eb-9ab4-08c2c2af183a.png" width="1000" height="200"/>

__Figure 3__: Training curves for the four selected models

## Model Evaluation ##

The Receiver Operating Chracteristic (ROC) curve in Figure 4 which shows the True Positive Rate against the False Positive Rate as the discrimination threshold is varied. This demonstrates that a high diagnostic ability of all classifiers. The MobileNetV2 model is able to achieve 97% accuracy while maintaining a high sensitivity.

<img src="https://user-images.githubusercontent.com/5468707/124141549-ca0a9d00-da89-11eb-9483-ed4385bcb84e.png" width="500" height="500"/>

__Figure 4__: Receiver Operating Chracteristic (ROC) curve  of the four chosen models
