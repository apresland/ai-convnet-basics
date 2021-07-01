import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Activation

class DenseNet169Wrapper(tf.keras.Model):

    def __init__(self, classes=1, **kwargs):
        super(DenseNet169Wrapper, self).__init__()      
        self.densnet = tf.keras.applications.DenseNet169(
            input_shape=(96,96,3),
            include_top=False,
            weights=None)
        self.global_max_pooling_2d = GlobalMaxPooling2D()
        self.global_avg_pooling_2d = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.concatenate = Concatenate(axis=-1)
        self.dropout = Dropout(0.5)
        self.dense = Dense(256, activation='relu')
        self.classifier = Dense(classes, activation='sigmoid')


    def call(self, inputs, **kwargs):
        x = self.densnet(inputs)
        x1 = self.global_max_pooling_2d(x)
        x2 = self.global_avg_pooling_2d(x)
        x3 = self.flatten(x)
        x = self.concatenate([x1,x2,x3])
        x = self.dropout(x)
        x = self.dense(x)
        return self.classifier(x)