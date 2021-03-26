import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras import optimizers
from tensorflow.keras.applications import ResNet50


class ResNet50Wrapper(tf.keras.Model):
    def __init__(self, classes=1, **kwargs):
        super(ResNet50Wrapper, self).__init__()
        self.resnet = ResNet50(
            weights=None,
            input_shape=(96,96,3),
            include_top=False)
        self.global_max_pooling_2d = GlobalMaxPooling2D()
        self.global_avg_pooling_2d = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.concatenate = Concatenate(axis=-1)
        self.dropout = Dropout(0.5)
        self.dense = Dense(512, activation='relu')
        self.classifier = Dense(classes, activation='sigmoid')

    def call(self, inputs, **kwargs):
        x = self.resnet(inputs)
        x1 = self.global_max_pooling_2d(x)
        x2 = self.global_avg_pooling_2d(x)
        x3 = self.flatten(x)
        x = self.concatenate([x1,x2,x3])
        x = self.dense(x)
        x = self.dropout(x)
        return self.classifier(x)