# imported from: https://github.com/raghakot/keras-resnet

from __future__ import division

import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K

# Added
import numpy as np
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from keras import optimizers
from keras.utils import np_utils


def _bn_relu(input):
    """Helper to build a BN -> relu block
    """
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)


def _conv_bn_relu(**conv_params):
    """Helper to build a conv -> BN -> relu block
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        conv = Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu(conv)

    return f


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _shortcut(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"
    """
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    input_shape = K.int_shape(input)
    residual_shape = K.int_shape(residual)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
        shortcut = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                          kernel_size=(1, 1),
                          strides=(stride_width, stride_height),
                          padding="valid",
                          kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block(block_function, filters, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks.
    """
    def f(input):
        for i in range(repetitions):
            init_strides = (1, 1)
            if i == 0 and not is_first_layer:
                init_strides = (2, 2)
            input = block_function(filters=filters, init_strides=init_strides,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Basic 3 X 3 convolution blocks for use on resnets with layers <= 34.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv1 = Conv2D(filters=filters, kernel_size=(3, 3),
                           strides=init_strides,
                           padding="same",
                           kernel_initializer="he_normal",
                           kernel_regularizer=l2(1e-4))(input)
        else:
            conv1 = _bn_relu_conv(filters=filters, kernel_size=(3, 3),
                                  strides=init_strides)(input)

        residual = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv1)
        return _shortcut(input, residual)

    return f


def bottleneck(filters, init_strides=(1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer resnet.
    Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
    Returns:
        A final conv layer of filters * 4
    """
    def f(input):

        if is_first_block_of_first_layer:
            # don't repeat bn->relu since we just did bn->relu->maxpool
            conv_1_1 = Conv2D(filters=filters, kernel_size=(1, 1),
                              strides=init_strides,
                              padding="same",
                              kernel_initializer="he_normal",
                              kernel_regularizer=l2(1e-4))(input)
        else:
            conv_1_1 = _bn_relu_conv(filters=filters, kernel_size=(1, 1),
                                     strides=init_strides)(input)

        conv_3_3 = _bn_relu_conv(filters=filters, kernel_size=(3, 3))(conv_1_1)
        residual = _bn_relu_conv(filters=filters * 4, kernel_size=(1, 1))(conv_3_3)
        return _shortcut(input, residual)

    return f


def _handle_dim_ordering():
    global ROW_AXIS
    global COL_AXIS
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        ROW_AXIS = 1
        COL_AXIS = 2
        CHANNEL_AXIS = 3
    else:
        CHANNEL_AXIS = 1
        ROW_AXIS = 2
        COL_AXIS = 3


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(units=num_outputs, kernel_initializer="he_normal",
                      activation="softmax")(flatten1)

        model = Model(inputs=input, outputs=dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])

class ResnetBuilderMultiLoss(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a custom ResNet like architecture.
        Args:
            input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)
            num_outputs: The number of outputs at final softmax layer
            block_fn: The block function to use. This is either `basic_block` or `bottleneck`.
                The original paper used basic_block for layers < 50
            repetitions: Number of repetitions of various block units.
                At each block unit, the number of filters are doubled and the input size is halved
        Returns:
            The keras `Model`.
        """
        _handle_dim_ordering()
        if len(input_shape) != 3:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols)")

        # Permute dimension order if necessary
        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[0])

        # Load function from str if needed.
        block_fn = _get_block(block_fn)

        input = Input(shape=input_shape)
        conv1 = _conv_bn_relu(filters=64, kernel_size=(7, 7), strides=(2, 2))(input)
        pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(conv1)

        block = pool1
        filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, filters=filters, repetitions=r, is_first_layer=(i == 0))(block)
            filters *= 2

        # Last activation
        block = _bn_relu(block)

        # Classifier block
        block_shape = K.int_shape(block)
        pool2 = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                                 strides=(1, 1))(block)
        flatten1 = Flatten()(pool2)

        all_dense = list()
        for k, v in num_outputs.items():
            dense = Dense(units=v, kernel_initializer="he_normal",
                          activation="softmax", name=k)(flatten1)
            all_dense.append(dense)

        model = Model(inputs=input, outputs=all_dense)
        return model

    @staticmethod
    def build_resnet_18(input_shape, num_outputs):
        return ResnetBuilderMultiLoss.build(input_shape, num_outputs, basic_block, [2, 2, 2, 2])

    @staticmethod
    def build_resnet_34(input_shape, num_outputs):
        return ResnetBuilderMultiLoss.build(input_shape, num_outputs, basic_block, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_50(input_shape, num_outputs):
        return ResnetBuilderMultiLoss.build(input_shape, num_outputs, bottleneck, [3, 4, 6, 3])

    @staticmethod
    def build_resnet_101(input_shape, num_outputs):
        return ResnetBuilderMultiLoss.build(input_shape, num_outputs, bottleneck, [3, 4, 23, 3])

    @staticmethod
    def build_resnet_152(input_shape, num_outputs):
        return ResnetBuilderMultiLoss.build(input_shape, num_outputs, bottleneck, [3, 8, 36, 3])


class Resnet:
    def __init__(self, input_shape, num_classes, resnet_mode='resnet_18', gpu=0):
        resnet_modes = {
            'resnet_18': ResnetBuilder.build_resnet_18,
            'resnet_34': ResnetBuilder.build_resnet_34,
            'resnet_50': ResnetBuilder.build_resnet_50,
            'resnet_101': ResnetBuilder.build_resnet_101,
            'resnet_152': ResnetBuilder.build_resnet_152
        }

        if resnet_mode not in resnet_modes:
            print('Incorrect resnet mode: %s' % resnet_mode)
            print('Available resnet modes are: %s' %(str(list(resnet_modes.keys()))))

            return

        with K.tf.device('/gpu:' + str(gpu)):
            m = resnet_modes[resnet_mode](input_shape=input_shape, num_outputs=num_classes)
        sgd = optimizers.SGD(lr=0.01)
        adam = optimizers.Adam()
        m.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

        self.model = m

    def test(self, x_train, y_train, x_test, y_test, verbose=1):
        print('##########')
        print('Resnet Test')
        print('##########')
        # early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        # self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=64,
        #                callbacks=['early_stopping'])
        self.model.fit(x_train, y_train, validation_split=0.1, epochs=50, batch_size=64,
                       verbose=verbose,
                       # callbacks=[early_stopping],
                       shuffle=True)
        # loss, metrics = self.model.evaluate(x=x_test, y=y_test, batch_size=64)
        # print(metrics)
        _y = self.model.predict(x=x_test, batch_size=128)
        _y = np.argmax(_y, axis=1)
        y_test = np.argmax(y_test, axis=1)

        accuracy = [y_test == _y][0]
        accuracy = float(len(accuracy[accuracy==True]) / len(y_test))
        print('##########')
        print("Accuracy: %.4f" % (accuracy))
        print('##########')
        print(classification_report(y_test, _y))


class ResnetMultiLoss:
    def __init__(self, input_shape, num_classes, resnet_mode='resnet_18', gpu=0):
        resnet_modes = {
            'resnet_18': ResnetBuilderMultiLoss.build_resnet_18,
            'resnet_34': ResnetBuilderMultiLoss.build_resnet_34,
            'resnet_50': ResnetBuilderMultiLoss.build_resnet_50,
            'resnet_101': ResnetBuilderMultiLoss.build_resnet_101,
            'resnet_152': ResnetBuilderMultiLoss.build_resnet_152
        }

        if resnet_mode not in resnet_modes:
            print('Incorrect resnet mode: %s' % resnet_mode)
            print('Available resnet modes are: %s' % (str(list(resnet_modes.keys()))))

            return

        with K.tf.device('/gpu:' + str(gpu)):
            m = resnet_modes[resnet_mode](input_shape=input_shape, num_outputs=num_classes)
        sgd = optimizers.SGD(lr=0.01)
        adam = optimizers.Adam()
        losses = {'amusement': 'categorical_crossentropy',
                  'immersion': 'categorical_crossentropy',
                  'difficulty': 'categorical_crossentropy',
                  'emotion': 'categorical_crossentropy'}
        loss_weights = {'amusement': 1.0, 'immersion': 1.0, 'difficulty': 1.0, 'emotion': 1.0}
        m.compile(loss=losses, loss_weights=loss_weights, optimizer=adam, metrics=['accuracy'])

        self.model = m

    def test(self, x_train, y_train, x_test, y_test, verbose=1):
        print('##########')
        print('Resnet Test')
        print('##########')
        # early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
        # self.model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=50, batch_size=64,
        #                callbacks=['early_stopping'])
        label_names = ['amusement', 'immersion', 'difficulty', 'emotion']

        self.model.fit(x_train, {'amusement': np_utils.to_categorical(y_train[0]),
                                 'immersion': np_utils.to_categorical(y_train[1]),
                                 'difficulty': np_utils.to_categorical(y_train[2]),
                                 'emotion': np_utils.to_categorical(y_train[3])},
                       validation_data=(x_test, {'amusement': np_utils.to_categorical(y_test[0]),
                                                 'immersion': np_utils.to_categorical(y_test[1]),
                                                 'difficulty': np_utils.to_categorical(y_test[2]),
                                                 'emotion': np_utils.to_categorical(y_test[3])}), epochs=1,
                       batch_size=64,
                       verbose=verbose, shuffle=True, class_weight={
                'amusement': class_weight.compute_class_weight('balanced',
                                                               np.unique(y_train[0]),
                                                               y_train[0]),
                'immersion': class_weight.compute_class_weight('balanced',
                                                               np.unique(y_train[1]),
                                                               y_train[1]),
                'difficulty': class_weight.compute_class_weight('balanced',
                                                                np.unique(y_train[2]),
                                                                y_train[2]),
                'emotion': class_weight.compute_class_weight('balanced',
                                                             np.unique(y_train[3]),
                                                             y_train[3])

            })
        # loss, metrics = self.model.evaluate(x=x_test, y=y_test, batch_size=64)
        # print(metrics)
        y_pred = self.model.predict(x=x_test, batch_size=128)
        #         _y = np.argmax(_y, axis=1)
        #         y_test = np.argmax(y_test, axis=1)

        #         accuracy = [y_test == _y][0]
        #         accuracy = float(len(accuracy[accuracy==True]) / len(y_test))
        #         print('##########')
        #         print("Accuracy: %.4f" % (accuracy))
        #         print('##########')
        #         print(classification_report(y_test, _y))

        _y_final = list()
        for idx, _y in enumerate(y_pred):
            print('##########')
            print(label_names[idx])
            _y = np.argmax(_y, axis=1)
            _y_final.append(_y)
            _y_test = y_test[idx]

            accuracy = [_y_test == _y][0]
            accuracy = float(len(accuracy[accuracy == True]) / len(_y_test))

            print("Accuracy: %.4f" % (accuracy))
            print('##########')
            print(classification_report(_y_test, _y))

        return y_test, _y_final

