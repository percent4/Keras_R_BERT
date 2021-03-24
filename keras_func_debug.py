# -*- coding: utf-8 -*-
# @Time : 2021/3/24 11:44
# @Author : Jclian91
# @File : keras_func_debug.py
# @Place : Yangpu, Shanghai
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Add, Lambda, Multiply, Average, average, Flatten
from keras.models import Model
import numpy as np


if __name__ == '__main__':

    batch_size = 2

    a = K.constant([[0, 1, 1, 0],
                    [1, 0, 0, 1]])
    b = np.random.random(size=(batch_size, 4, 3))
    print(b)
    b = K.constant(b)
    c = K.batch_dot(a, b, axes=1)

    with tf.Session() as sess:
        output_array = sess.run(c)
        print(output_array)
        print(a.shape, b.shape, output_array.shape)

