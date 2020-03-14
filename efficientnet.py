# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 18:00:52 2019

@author: Oliver Lin
The Best or nothing!!!
"""

from keras.layers import Input, Dense, Flatten, ReLU, Add, Multiply, Lambda, Activation, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, DepthwiseConv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import glorot_uniform, RandomNormal, Zeros, Initializer
from keras.optimizers import Adam
from keras.optimizers import RMSprop
from keras.layers import BatchNormalization, Activation, Dropout
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import keras.layers
import tensorflow as tf
import numpy as np
import pandas as pd


class EfficientNetCovInitializer(Initializer):
    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        kernel_height, kernel_width, _, out_filters = shape
        fan_out = int(kernel_height * kernel_width * out_filters)
        return tf.random_normal(
            shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)

    pass


class EfficientNetDenseInitializer(Initializer):
    """Initialization for dense kernels.
        This initialization is equal to
          tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                          distribution='uniform').
        It is written out explicitly base_path for clarity.
        # Arguments:
          shape: shape of variable
          dtype: dtype of variable
          partition_info: unused
        # Returns:
          an initialization for the variable
    """

    def __init__(self):
        super(EfficientNetDenseInitializer, self).__init__()

    def __call__(self, shape, dtype=None):
        dtype = dtype or K.floatx()

        init_range = 1.0 / np.sqrt(shape[1])
        return tf.random_uniform(shape, -init_range, init_range, dtype=dtype)


class Swish(keras.layers.Layer):
    def call(self, inputs):
        return tf.nn.swish(inputs)


def SEBlock(input_filters, expand_factor, se_ratio):
    num_reduce_filters = max(1, int(input_filters * se_ratio))

    #    filters = input_filters * expand_factor
    #    print(num_reduce_filters)
    #    print(filters)
    def block(inputs):
        X = inputs
        # global average pooling
        X = Lambda(lambda a: K.mean(a, axis=[1, 2], keepdims=True))(X)
        X = ConvBlock(input_filters=num_reduce_filters, kernel_size=[1, 1], strides=[1, 1], use_bias=True,
                      use_batchnorm=False)(X)
        X = ConvBlock(input_filters=input_filters, kernel_size=[1, 1], strides=[1, 1], use_bias=True,
                      use_batchnorm=False, use_swish=False)(X)
        X = Activation('sigmoid')(X)
        # each channel feature map weight
        X = Multiply()([X, inputs])
        #        print(inputs.shape)
        return X

    return block
    pass


def DWConvBlock(kernel_size, strides):
    def block(X):
        X = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='SAME', depth_multiplier=1,
                            depthwise_initializer=EfficientNetCovInitializer())(X)
        X = BatchNormalization(axis=3)(X)
        X = Swish()(X)
        return X

    return block
    pass


def ConvBlock(input_filters, kernel_size, strides, use_batchnorm=True, use_swish=True, use_bias=False):
    def block(inputs):
        X = inputs
        X = Conv2D(filters=input_filters, kernel_size=kernel_size, strides=strides, padding='SAME', use_bias=use_bias,
                   kernel_initializer=EfficientNetCovInitializer())(X)
        if (use_batchnorm):
            X = BatchNormalization(axis=3)(X)
        if (use_swish):
            X = Swish()(X)
        return X

    return block
    # relu BNs
    pass


def MBConvBlock(input_filters, output_filters, kernel_size, strides, expand_factor, se_ratio, drop_connect_rate=None):
    has_se = (se_ratio is not None) and (se_ratio > 0) and (se_ratio <= 1)
    filters = input_filters * expand_factor

    def block(inputs):
        if (expand_factor != 1):
            X = ConvBlock(filters, kernel_size=[1, 1], strides=[1, 1], use_bias=False)(inputs)
            pass
        else:
            X = inputs
            pass

        X = DWConvBlock(kernel_size=kernel_size, strides=strides)(X)

        if (has_se):
            X = SEBlock(input_filters=filters, expand_factor=expand_factor, se_ratio=se_ratio)(X)
            pass

        X = ConvBlock(output_filters, kernel_size=[1, 1], strides=[1, 1], use_bias=False, use_swish=False)(X)

        if (all(s == 1 for s in strides) and (input_filters == output_filters)):
            X = Add()([inputs, X])
            pass
        return X

    return block
    pass


def relu6(X):
    X = ReLU(max_value=6)(X)
    return X
    pass


def EfficientNet(input_shape):
    inputs = Input(input_shape)
    block_list = [
        {'input_filters': 32, 'output_filters': 16, 'kernel_size': [3, 3], 'strides': [1, 1], 'num_repeat': 1,
         'se_ratio': 0.25, 'expand_factor': 1},
        {'input_filters': 16, 'output_filters': 24, 'kernel_size': [3, 3], 'strides': [2, 2], 'num_repeat': 2,
         'se_ratio': 0.25, 'expand_factor': 6},
        {'input_filters': 24, 'output_filters': 40, 'kernel_size': [3, 3], 'strides': [2, 2], 'num_repeat': 2,
         'se_ratio': 0.25, 'expand_factor': 6},
        {'input_filters': 40, 'output_filters': 80, 'kernel_size': [3, 3], 'strides': [2, 2], 'num_repeat': 3,
         'se_ratio': 0.25, 'expand_factor': 6},
        {'input_filters': 82, 'output_filters': 112, 'kernel_size': [3, 3], 'strides': [1, 1], 'num_repeat': 3,
         'se_ratio': 0.25, 'expand_factor': 6},
        {'input_filters': 112, 'output_filters': 192, 'kernel_size': [3, 3], 'strides': [2, 2], 'num_repeat': 4,
         'se_ratio': 0.25, 'expand_factor': 6},
        {'input_filters': 192, 'output_filters': 320, 'kernel_size': [3, 3], 'strides': [1, 1], 'num_repeat': 1,
         'se_ratio': 0.25, 'expand_factor': 6}
    ]

    X = inputs
    X = ConvBlock(input_filters=32, kernel_size=[3, 3], strides=[2, 2], use_bias=False)(X)
    for block_index, block_args in enumerate(block_list):

        X = MBConvBlock(block_args['input_filters'], block_args['output_filters'], block_args['kernel_size'],
                        block_args['strides'], block_args['expand_factor'], \
                        block_args['se_ratio'])(X)
        if (block_args['num_repeat'] > 1):
            for _ in range(block_args['num_repeat'] - 1):
                X = MBConvBlock(block_args['input_filters'], block_args['output_filters'], block_args['kernel_size'],
                                [1, 1], block_args['expand_factor'], \
                                block_args['se_ratio'])(X)
            pass
    X = ConvBlock(input_filters=1280, kernel_size=[1, 1], strides=[1, 1], use_bias=False)(X)
    X = GlobalAveragePooling2D()(X)
    X = Dense(3, activation='softmax', kernel_initializer=EfficientNetDenseInitializer())(X)
    outputs = X
    return Model(inputs=inputs, outputs=outputs)


train_dir = './train'
test_dir = './test'
label_dir = './train_labels.csv'
img_shape = (128, 173)


def input_loads():
    df = pd.read_csv(label_dir)
    df['file_id'] = df['file_id'].astype('str')
    df['file_id'] = df['file_id'].apply(lambda x: x + '.png')

    df['accent'] = df['accent'].astype('str')
    # df['accent'] = df['accent'].map(onehot)

    # label = pd.DataFrame(df['accent'].map(onehot).tolist(), columns=['0', '1', '2'])
    # df = pd.concat([df['file_id'], label], axis=1)
    print(df.head())

    datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=train_dir,
        x_col='file_id',
        y_col='accent',
        class_mode='categorical',
        target_size=(128, 174),
        batch_size=32
    )
    return train_generator


def runmodel(model):
    model.fit_generator(
        generator=input_loads(),
        steps_per_epoch=100,
        epochs=20,
        verbose=1,
    )
    f_names = ['./test/%d.png' % i for i in range(20000, 25377)]

    img = []
    result = []
    for i in range(len(f_names)):
        images = load_img(f_names[i], target_size=(128, 174))
        x = img_to_array(images)
        x = np.expand_dims(x, axis=0)
        y = model.predict(x)
        result.append(y.tolist()[0])
        print('loading no.%s image' % i)

    a = [result[i].index(max(result[i])) for i in range(len(result))]
    test_id = pd.DataFrame(pd.read_csv('submission_format.csv'))
    a = pd.concat([test_id['file_id'], pd.DataFrame(a, columns=['accent'])['accent']], axis=1)
    pd.DataFrame(a).to_csv('result.csv')
    print('\nfinish`````````````````````````````')


def predict_img():
    datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = datagen.flow_from_directory(
        directory=test_dir,
        target_size=(128, 174),
        batch_size=10,
    )
    return test_generator


model = EfficientNet((128, 174, 3))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(1e-4),
              metrics=['accuracy'])
print(model.summary())
runmodel(model)
