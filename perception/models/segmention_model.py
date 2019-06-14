"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

import os
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.merge import add, multiply
from tensorflow.python.keras.layers import Lambda, Input, Conv2D, Conv2DTranspose, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, core, Dropout, normalization, concatenate, Activation
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.core import Layer, InputSpec
from tensorflow.python.keras.layers.advanced_activations import LeakyReLU
from tensorflow.python.keras.utils import plot_model

from perception.bases.model_base import ModelBase


class SegmentionModel(ModelBase):

    def __init__(self, config=None):
        super(SegmentionModel, self).__init__(config)
        self.patch_height = config.patch_height
        self.patch_width = config.patch_width
        self.num_seg_class = config.seg_num

        self.build_model()
        self.save()

    def encoding_block(self, filters, strides=(1, 1), padding='same'):
        def f(inputs):
            conv = Conv2D(filters, strides, padding=padding)(inputs)
            conv = normalization.BatchNormalization(
                epsilon=2e-05, axis=3, momentum=0.9, weights=None, beta_initializer='RandomNormal', gamma_initializer='one')(conv)
            conv = LeakyReLU(alpha=0.3)(conv)
            conv = Dropout(0.2)(conv)
            conv = Conv2D(filters, strides, dilation_rate=2, padding=padding)(conv)
            conv = LeakyReLU(alpha=0.3)(conv)
            conv = Conv2D(filters, strides, dilation_rate=4, padding=padding)(conv)
            conv = LeakyReLU(alpha=0.3)(conv)
            return conv
        return f

    def decoding_block(self, filters, strides=(1, 1), padding='same'):
        def f(up, conv):
            concat = concatenate([up, conv], axis=-1)
            conv = Conv2D(filters, strides, padding=padding)(concat)
            conv = LeakyReLU(alpha=0.3)(conv)
            conv = Dropout(0.2)(conv)
            conv = Conv2D(filters, strides, padding=padding)(conv)
            conv = LeakyReLU(alpha=0.3)(conv)
            return conv
        return f

    def se_block(self, ratio):
        def f(inputs):
            squeeze = GlobalAveragePooling2D()(inputs)
            out_dim = squeeze.get_shape().as_list()[-1]
            excitation = Dense(units=out_dim / ratio)(squeeze)
            excitation = Activation("relu")(excitation)
            excitation = Dense(units=out_dim)(excitation)
            excitation = Activation("sigmoid")(excitation)
            excitation = Reshape([1, 1, out_dim])(excitation)
            scale = Multiply()([inputs, excitation])
            return scale
        return f

    def build_model(self):
        inputs = Input((self.patch_height, self.patch_width, 1))
        conv1 = self.encoding_block(32, strides=(3, 3), padding='same')(inputs)
        conv1 = self.se_block(ratio=2)(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = self.encoding_block(64, strides=(3, 3), padding='same')(pool1)
        conv2 = self.se_block(ratio=2)(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = self.encoding_block(128, strides=(3, 3), padding='same')(pool2)
        conv3 = self.se_block(ratio=2)(conv3)

        up = UpSampling2D(size=(2, 2))(conv3)
        conv4 = decoding_block(filters, strides=(3, 3), padding='same')(up, conv2)
        conv4 = self.se_block(ratio=2)(conv4)
        up1 = UpSampling2D(size=(2, 2))(conv4)
        conv5 = decoding_block(filters, strides=(3, 3), padding='same')(up1, conv1)
        conv5 = self.se_block(ratio=2)(conv5)

        conv6 = Conv2D(self.num_seg_class + 1, (1, 1), padding='same')(conv5)
        conv6 = LeakyReLU(alpha=0.3)(conv6)
        conv6 = core.Reshape((self.patch_height * self.patch_width, self.num_seg_class + 1))(conv6)

        act = Activation('softmax')(conv6)

        model = Model(inputs=inputs, outputs=act)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
        self.model = model
