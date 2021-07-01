"""
The script evaluates models on testing data .
needed for selection of models for the final prediction task.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
#from utils.utils import *
#from utils import factory
from data.data_generator import DataGenerator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.args import get_args
#import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

import numpy as np
import os
import argparse
from tensorflow.keras.models import load_model
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches

preprocess_input = tf.keras.applications.resnet50.preprocess_input

class GradCAM:
    # Adapted with some modification from https://www.pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/
    def __init__(self, model, layerName=None):
        """
        model: pre-softmax layer (logit layer)
        """
        self.model = model
        self.layerName = layerName
            
        if self.layerName == None:
            self.layerName = self.find_target_layer()
    
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM")
            
    def compute_heatmap(self, image, classIdx, upsample_size, eps=1e-5):
        gradModel = tf.keras.Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output, self.model.output]
        )
        # record operations for automatic differentiation
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOuts, preds) = gradModel(inputs) # preds after softmax
            loss = preds[:,classIdx]
        
        # compute gradients with automatic differentiation
        grads = tape.gradient(loss, convOuts)
        # discard batch
        convOuts = convOuts[0]
        grads = grads[0]
        norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(eps))
        
        # compute weights
        weights = tf.reduce_mean(norm_grads, axis=(0,1))
        cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1)
        
        # Apply reLU
        cam = np.maximum(cam, 0)
        cam = cam/np.max(cam)
        cam = cv.resize(cam, upsample_size,interpolation=cv.INTER_LINEAR)
        
        # convert to 3D
        cam3 = np.expand_dims(cam, axis=2)
        cam3 = np.tile(cam3, [1,1,3])
        
        return cam3
    
def overlay_gradCAM(img, cam3, prob):
    cam3 = np.uint8(255*cam3*prob)
    cam3 = cv.applyColorMap(cam3, cv.COLORMAP_JET)
    
    new_img = 0.3*cam3 + 0.5*img
    
    return (new_img*255.0/new_img.max()).astype("uint8")


@tf.custom_gradient
def guidedRelu(x):
    def grad(dy):
        return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
    return tf.nn.relu(x), grad

# Reference: https://github.com/eclique/keras-gradcam with adaption to tensorflow 2.0  
class GuidedBackprop:
    def __init__(self,model, layerName=None):
        self.model = model
        self.layerName = layerName
        self.gbModel = self.build_guided_model()
        
        if self.layerName == None:
            self.layerName = self.find_target_layer()

    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply Guided Backpropagation")

    def build_guided_model(self):
        gbModel = tf.keras.Model(
            inputs = [self.model.inputs],
            outputs = [self.model.get_layer(self.layerName).output]
        )
        layer_dict = [layer for layer in gbModel.layers[1:] if hasattr(layer,"activation")]
        for layer in layer_dict:
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
        
        return gbModel
    
    def guided_backprop(self, images, upsample_size):
        """Guided Backpropagation method for visualizing input saliency."""
        with tf.GradientTape() as tape:
            inputs = tf.cast(images, tf.float32)
            tape.watch(inputs)
            outputs = self.gbModel(inputs)

        grads = tape.gradient(outputs, inputs)[0]

        saliency = cv.resize(np.asarray(grads), upsample_size)

        return saliency

def deprocess_image(x):
    """Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    """
    # normalize tensor: center on 0., ensure std is 0.25
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def show_gradCAMs(model, gradCAM, guidedBP, im_path):
    """
    model: softmax layer
    """
    plt.subplots(figsize=(10, 10))

    img = cv.imread(im_path)
    upsample_size = (img.shape[1],img.shape[0])
    
    # Show original image
    plt.subplot(1,3,1)
    plt.imshow(cv.cvtColor(img,cv.COLOR_BGR2RGB))
    plt.title("Filename: {}".format(im_path), fontsize=20)
    plt.axis("off")
    
    # Show overlayed grad
    plt.subplot(1,3,2)
    im = img_to_array(load_img(os.path.join(im_path), target_size=(96,96)))
    x = np.expand_dims(im, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    print('probability', preds[0])
    idx = preds.argmax()
    cam3 = gradCAM.compute_heatmap(image=x, classIdx=idx, upsample_size=upsample_size)
    new_img = overlay_gradCAM(img, cam3, preds[0])
    new_img = cv.cvtColor(new_img, cv.COLOR_BGR2RGB)
    plt.imshow(new_img)
    #plt.title("GradCAM - Pred: {}. Prob: {}".format(res[0],res[1]), fontsize=20)
    plt.axis("off")
        
    # Show guided GradCAM
    plt.subplot(1,3,3)
    gb = guidedBP.guided_backprop(x, upsample_size)
    guided_gradcam = deprocess_image(gb*cam3)
    guided_gradcam = cv.cvtColor(guided_gradcam, cv.COLOR_BGR2RGB)
    plt.imshow(guided_gradcam)
    plt.title("Guided GradCAM", fontsize=20)
    plt.axis("off")
        
    plt.show()


def main():
        
    if tf.test.gpu_device_name(): 
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
       print("Please install GPU version of TF")

    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        config = process_config(args.config)
    except:
        print("missing or invalid arguments")
        exit(0)

    data = DataGenerator(config)

    checkpoint = 'resnet50-01-0.8267.h5'

    #evaluate model on test data  
    model_path = os.path.join('./experiments/2021-03-20/densnet169/checkpoints','densnet169-12-0.9396.h5')
    print('Loading model..',model_path)
    model =  load_model(model_path)
    print('evaluating model..')
    loss, accuracy = model.evaluate_generator(data.valid_datagen, 400)
    print('--Test: Acc {} Loss: {} -- Config Model: {}'.format(loss,accuracy, checkpoint))

    model_logit = tf.keras.Model(model.input,model.layers[-2].output)
    gradCAM = GradCAM(model=model_logit, layerName="conv5_block3_out")
    guidedBP = GuidedBackprop(model=model, layerName="conv5_block3_out")

    img_path = '/data/patchcamelyon/staged/test/positive/7f13b65a86ca6331ae976735cf60633cbdd58321.tif'
    show_gradCAMs(model, gradCAM, guidedBP, img_path)

    img_path = '/data/patchcamelyon/staged/test/negative/41b654b9184b568ae3848fac826d708abcac359f.tif'
    show_gradCAMs(model, gradCAM, guidedBP, img_path)

if __name__ == '__main__':
    main()