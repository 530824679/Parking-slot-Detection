# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : basenet.py
# Description :base operators.
# --------------------------------------

import tensorflow as tf

def bn(input):
    with tf.variable_scope('bn'):
        gamma=tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='weight',trainable=True)
        beta = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='bias',trainable=True)
        mean = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='running_mean',trainable=True)
        var = tf.Variable(tf.random_normal(shape=[input.shape[-1].value]), name='running_var',trainable=True)

        out = tf.nn.batch_normalization(input, mean, var, beta, gamma,variance_epsilon=0.001)
        return out

def conv(input, out_channels, ksize, stride, name='conv', add_bias=False):
    filter = tf.Variable(tf.random_normal(shape=[ksize, ksize, input.shape[-1].value, out_channels]), name=name+'/weight', trainable=True)

    if ksize > 1:
        pad_h, pad_w = ksize//2, ksize//2
        paddings = tf.constant([[0, 0], [pad_h, pad_h], [pad_w, pad_w], [0, 0]])
        input = tf.pad(input, paddings, 'CONSTANT')
    net = tf.nn.conv2d(input, filter, [1,stride, stride, 1], padding="VALID")

    if add_bias:
        bias = tf.Variable(tf.random_normal(shape=[out_channels]), name=name + '/bias',trainable=True)
        net = tf.nn.bias_add(net,bias)
    return net

def convBnLeakly(input, out_channels, ksize, stride, name):
    with tf.variable_scope(name):
        net = conv(input, out_channels, ksize, stride)
        net = bn(net)
        # swish
        # net=tf.nn.sigmoid(net)*net

        # v2.0
        # net=tf.nn.leaky_relu(net,alpha=0.1)

        # v3.0
        net = net * tf.nn.relu6(net + 3.0) / 6.0

        return net

def focus(input, out_channels, ksize, name):
    s1 = input[:, ::2, ::2, :]
    s2 = input[:, 1::2, ::2, :]
    s3 = input[:, ::2, 1::2, :]
    s4 = input[:, 1::2, 1::2, :]

    net = tf.concat([s1, s2, s3, s4], axis=-1)
    net = convBnLeakly(net, out_channels, ksize, 1, name+'/conv')
    return net

def bottleneck(input, c1, c2, shortcut, e, name):
    with tf.variable_scope(name):
        net = convBnLeakly(input,int(c2 * e), 1, 1, 'cv1')
        net = convBnLeakly(net, c2, 3, 1, 'cv2')

        if (shortcut and c1==c2):
            net += input
        return net

def bottleneckCSP(input, c1, c2, n, shortcut, e, name):
    c_ = int(c2 * e)
    with tf.variable_scope(name):
        net1 = convBnLeakly(input, c_, 1, 1, 'cv1')
        for i in range(n):
            net1 = bottleneck(net1, c_, c_, shortcut, 1.0, name='m/%d'%i)

        net1 = conv(net1, c_, 1, 1, name='cv3')
        net2 = conv(input, c_, 1, 1, 'cv2')

        net = tf.concat((net1, net2), -1)
        net = bn(net)
        net = tf.nn.leaky_relu(net, alpha=0.1)

        net = convBnLeakly(net, c2, 1, 1, 'cv4')
        return net

def spp(input, c1, c2, k1, k2, k3, name):
    c_ = c1//2
    with tf.variable_scope(name):
        net = convBnLeakly(input, c_, 1, 1, 'cv1')

        net1 = tf.nn.max_pool(net, ksize=[1, k1, k1, 1], strides=[1, 1, 1, 1], padding="SAME")
        net2 = tf.nn.max_pool(net, ksize=[1, k2, k2, 1], strides=[1, 1, 1, 1], padding="SAME")
        net3 = tf.nn.max_pool(net, ksize=[1, k3, k3, 1], strides=[1, 1, 1, 1], padding="SAME")

        net = tf.concat((net, net1, net2, net3), -1)
        net = convBnLeakly(net, c2, 1, 1, 'cv2')

        return net