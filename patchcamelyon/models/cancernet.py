
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Activation, GlobalAveragePooling2D, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import MaxPooling2D

class CancerNet(tf.keras.Model):

    def __init__(self, classes=1, **kwargs):
        super(CancerNet, self).__init__()

        chanDim = -1

        # CONV => RELU => POOL
        self.block1_conv = SeparableConv2D(32, (3, 3), padding="same")
        self.block1_act = Activation("relu")
        self.block1_norm = BatchNormalization(axis=chanDim)
        self.block1_pool = MaxPooling2D(pool_size=(2, 2))
        self.block1_drop = Dropout(0.25)

        # (CONV => RELU => POOL) * 2
        self.block2_conv1 = SeparableConv2D(64, (3, 3), padding="same")
        self.block2_act1 = Activation("relu")
        self.block2_norm1 = BatchNormalization(axis=chanDim)
        self.block2_conv2 = SeparableConv2D(64, (3, 3), padding="same")
        self.block2_act2 = Activation("relu")
        self.block2_norm2 = BatchNormalization(axis=chanDim)
        self.block2_pool = MaxPooling2D(pool_size=(2, 2))
        self.block2_drop = Dropout(0.25)

        # (CONV => RELU => POOL) * 3
        self.block3_conv1 = SeparableConv2D(128, (3, 3), padding="same")
        self.block3_act1 = Activation("relu")
        self.block3_norm1 = BatchNormalization(axis=chanDim)
        self.block3_conv2 = SeparableConv2D(128, (3, 3), padding="same")
        self.block3_act2 = Activation("relu")
        self.block3_norm2 = BatchNormalization(axis=chanDim)
        self.block3_conv3 = SeparableConv2D(128, (3, 3), padding="same")
        self.block3_act3 = Activation("relu")
        self.block3_norm3 = BatchNormalization(axis=chanDim)
        self.block3_pool = MaxPooling2D(pool_size=(2, 2))
        self.block3_drop = Dropout(0.25)


        self.global_max_pooling_2d = GlobalMaxPooling2D()
        self.global_avg_pooling_2d = GlobalAveragePooling2D()
        self.flatten = Flatten()
        self.concatenate = Concatenate(axis=-1)
        self.dropout = Dropout(0.5)
        self.dense = Dense(256, activation='relu')

        # softmax classifier
        self.block5_dense = Dense(classes)
        self.block5_classifier = Activation("sigmoid")


    def call(self, inputs, **kwargs):


        # CONV => RELU => POOL
        x = self.block1_conv(inputs)
        x = self.block1_act(x)
        x = self.block1_norm(x)
        x = self.block1_pool(x)
        x = self.block1_drop(x)

        # (CONV => RELU => POOL) * 2
        x = self.block2_conv1(x)
        x = self.block2_act1(x)
        x = self.block2_norm1(x)
        x = self.block2_conv2(x)
        x = self.block2_act2(x)
        x = self.block2_norm2(x)
        x = self.block2_pool(x)
        x = self.block2_drop(x)

        # (CONV => RELU => POOL) * 3
        x = self.block3_conv1(x)
        x = self.block3_act1(x)
        x = self.block3_norm1(x)
        x = self.block3_conv2(x)
        x = self.block3_act2(x)
        x = self.block3_norm2(x)
        x = self.block3_conv3(x)
        x = self.block3_act3(x)
        x = self.block3_norm3(x)
        x = self.block3_pool(x)
        x = self.block3_drop(x)

        # first (and only) set of FC => RELU layers
        #x = self.block4_flatten(x)
        #x = self.block4_dense(x)
        #x = self.block4_act(x)
        #x = self.block4_norm(x)
        #x = self.block4_drop(x)

        x1 = self.global_max_pooling_2d(x)
        x2 = self.global_avg_pooling_2d(x)
        x = self.concatenate([x1,x2])
        x = self.dropout(x)
        x = self.dense(x)

        # softmax classifier
        x = self.block5_dense(x)
        return self.block5_classifier(x)