""" custom_function.py: Keras custom defined functions placed in this file """

__author__ = "Isaac Sim"
__copyright__ = "Copyright 2019, The Realtime EEG Analysis Project"
__credits__ = ["Isaac Sim"]
__license__ = ""
__version__ = "1.0.0"
__maintainer__ = ["Isaac Sim", "Dongjoon Jeon"]
__email__ = "gilgarad@igsinc.co.kr"
__status__ = "Development"

import tensorflow as tf
import keras.backend.tensorflow_backend as K


def ScoreActivationFromSigmoid(x, target_min=1, target_max=9):
    activated_x = K.sigmoid(x)
    return activated_x * (target_max - target_min) + target_min


def GetPadMask(q):
    mask = K.cast(K.not_equal(K.sum(q, axis=-1, keepdims=True), 0), 'float32')
    return mask


def GetCountNonZero(x):
    return 1 / tf.reduce_sum(tf.cast(x, 'float32'), axis=-2, keepdims=True)