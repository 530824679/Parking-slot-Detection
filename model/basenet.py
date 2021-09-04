# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : basenet.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, MaxPool2D

def autopad(kernel, padding=None):
    if padding is None:
        padding = kernel // 2 if isinstance(kernel, int) else [x // 2 for x in kernel]
    return padding

class SiLU(object):
    def __call__(self, x):
        return x * tf.nn.sigmoid(x)

class Mish(object):
    def __call__(self, x):
        return x * tf.math.tanh(tf.math.softplus(x))

class Conv(Layer):
    def __init__(self, filters, kernel_size, strides, padding=None, groups=1):
        super(Conv, self).__init__()
        self.conv = Conv2D(filters, kernel_size, strides, autopad(kernel_size, padding), groups=groups, use_bias=False,
                           kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                           kernel_regularizer=tf.keras.regularizers.L2(5e-4))
        self.bn = BatchNormalization()
        self.act = SiLU()

    def call(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(Layer):
    def __init__(self, filters, kernel_size, strides):
        super(DWConv, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, groups=1)

    def call(self, x):
        return self.conv(x)

class Focus(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding=None, groups=1):
        super(Focus, self).__init__()
        self.conv = Conv(filters, kernel_size, strides, padding, groups)

    def call(self, x):
        return self.conv(tf.concat([x[..., ::2, ::2, :],
                                    x[..., 1::2, ::2, :],
                                    x[..., ::2, 1::2, :],
                                    x[..., 1::2, 1::2, :]],
                                   axis=3))

class Upsample(Layer):
    def __init__(self, i=None, ratio=2, method='bilinear'):
        super(Upsample, self).__init__()
        self.ratio = ratio
        self.method = method

    def call(self, x):
        return tf.image.resize(x, (tf.shape(x)[1] * self.ratio, tf.shape(x)[2] * self.ratio), method=self.method)


class Concat(Layer):
    def __init__(self, dims=3):
        super(Concat, self).__init__()
        self.dims = dims

    def call(self, x):
        return tf.concat(x, self.dims)

class Bottleneck(Layer):
    def __init__(self, units, shortcut=True, groups=1, expansion=0.5):
        super(Bottleneck, self).__init__()
        c_ = int(units * expansion)
        self.conv1 = Conv(c_, 1, 1)
        self.conv2 = Conv(units, 3, 1, groups=groups)
        self.add = shortcut

    def call(self, x):
        if self.add:
            return x + self.conv2(self.conv1(x))
        return self.conv2(self.conv1(x))

class BottleneckCSP(Layer):
    def __init__(self, units, n_layer=1, shortcut=True, groups=1, expansion=0.5):
        super(BottleneckCSP, self).__init__()
        c_ = int(units * expansion)
        self.conv1 = Conv(c_, 1, 1)
        self.conv2 = Conv2D(c_, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv3 = Conv2D(c_, 1, 1, use_bias=False, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        self.conv4 = Conv(units, 1, 1)
        self.bn = BatchNormalization(momentum=0.03)
        self.act = Mish()
        self.modules = tf.keras.Sequential([Bottleneck(c_, shortcut, groups, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        y1 = self.conv3(self.modules(self.conv1(x)))
        y2 = self.conv2(x)
        return self.conv4(self.act(self.bn(tf.concat([y1, y2], axis=-1))))

class C3(Layer):
    def __init__(self, units, n_layer=1, shortcut=True, groups=1, expansion=0.5):
        super(C3, self).__init__()
        c_ = int(units * expansion)
        self.conv1 = Conv(c_, 1, 1)
        self.conv2 = Conv(c_, 1, 1)
        self.conv3 = Conv(units, 1, 1)
        self.modules = tf.keras.Sequential([Bottleneck(c_, shortcut, groups, expansion=1.0) for _ in range(n_layer)])

    def call(self, x):
        return self.cv3(tf.concat([self.modules(self.conv1(x)), self.conv2(x)], axis=3))

class SPP(Layer):
    def __init__(self, units, kernels=(5, 9, 13)):
        super(SPP, self).__init__()
        units_e = units // 2  # Todo:
        self.conv1 = Conv(units_e, 1, 1)
        self.conv2 = Conv(units, 1, 1)
        self.pool3 = MaxPool2D(pool_size=3, strides=1, padding=3 // 2)

    def call(self, x):
        x = self.conv1(x)
        return self.conv2(tf.concat([x] + [self.pool3(self.pool3(x)),
                                         self.pool3(self.pool3(self.pool3(self.pool3(x)))),
                                         self.pool3(self.pool3(self.pool3(self.pool3(self.pool3(self.pool3(x))))))], 3))
