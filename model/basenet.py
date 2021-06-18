# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : basenet.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf

initializer = tf.random_normal_initializer(stddev=0.01)
l2 = tf.keras.regularizers.l2(4e-5)

def conv(x, filters, kernel=1, stride=1, add_bias=False):
    if stride == 2:
        x = tf.keras.layers.ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    else:
        padding = 'same'

    x = tf.keras.layers.Conv2D(filters, kernel, stride, padding, use_bias=add_bias, kernel_initializer=initializer, kernel_regularizer=l2)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.03)(x)
    x = tf.keras.layers.Activation(tf.nn.swish)(x)

    return x

def bottleneck(x, filters, shortcut=True):
    inputs = x
    x = conv(x, filters, 1)
    x = conv(x, filters, 3)

    if shortcut:
        x = inputs + x
    return x

def csp(x, filters, n, shortcut=True):
    y = conv(x, filters // 2)
    for _ in range(n):
        y = bottleneck(y, filters // 2, shortcut)

    x = conv(x, filters // 2)
    x = tf.keras.layers.concatenate([x, y])

    x = conv(x, filters)
    return x

def spp(x, filters, k1, k2, k3, width, name):
    c_ = filters // 2
    with tf.variable_scope(name):
        x = conv(x, int(round(width * c_)), 1, 1)

        net1 = tf.nn.max_pool(x, [1, k1, k1, 1], [1, 1, 1, 1], padding="SAME")
        net2 = tf.nn.max_pool(x, [1, k2, k2, 1], [1, 1, 1, 1], padding="SAME")
        net3 = tf.nn.max_pool(x, [1, k3, k3, 1], [1, 1, 1, 1], padding="SAME")

        x = tf.keras.layers.concatenate([x, net1, net2, net3])
        x = conv(x, int(round(width * 1024)), 1, 1)

        return x