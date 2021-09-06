# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v5 network architecture
# --------------------------------------

import sys
import yaml
import math
import logging
import argparse
import numpy as np
import tensorflow as tf
from pathlib import Path
from copy import deepcopy

from utils.general import check_file, set_logging
from utils.tf_utils import select_device

from model.basenet import *
from utils.data_utils import *

# to run '$ python *.py' files in subdirectories
sys.path.append('./')
logger = logging.getLogger(__name__)

class Network(object):
    def __init__(self, cfg='odet.yaml', ch=3, num_classes=None, anchors=None):
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            self.yaml_file = Path(cfg).name
            with open(cfg) as f:
                self.yaml = yaml.load(f, Loader=yaml.SafeLoader)

        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if num_classes and num_classes != self.yaml['num_classes']:
            logger.info(f"Overriding model.yaml num_classes={self.yaml['num_classes']} with num_classes={num_classes}")
            self.yaml['num_classes'] = num_classes

        if anchors:
            logger.info(f'Overriding model.yaml anchors with anchors={anchors}')
            self.yaml['anchors'] = round(anchors)

        self.module_list, self.save = self.parse_model(deepcopy(self.yaml), ch=[ch])
        self.names_list = [str(i) for i in range(self.yaml['num_classes'])]

        # Build strides, anchors
        module = self.module_list[-1]
        if isinstance(module, Detect):
            module.anchors /= tf.reshape(module.stride, [-1, 1, 1])

    def __call__(self, img_size, name='odet'):
        x = tf.keras.Input([img_size, img_size, 3])
        output = self.forward(x)
        return tf.keras.Model(inputs=x, outputs=output, name=name)

    def forward(self, x):
        y = []
        for module in self.module_list:
            if module.f != -1:
                if isinstance(module.f, int):
                    x = y[module.f]
                else:
                    x = [x if j == -1 else y[j] for j in module.f]

            x = module(x)
            y.append(x)
        return x

    def parse_model(self, d, ch):
        logger.info('\n%3s%18s%3s%10s  %-40s%-30s' % ('', 'from', 'n', 'params', 'module', 'arguments'))
        anchors, num_classes, depth_multiple, width_multiple = d['anchors'], d['num_classes'], d['depth_multiple'], d['width_multiple']
        num_anchors = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        output_dims = num_anchors * (num_classes + 5)

        # layers, savelist, ch out
        layers, save, c2 = [], [], ch[-1]

        # from, number, module, args
        for i, (f, number, module, args) in enumerate(d['backbone'] + d['head']):
            module = eval(module) if isinstance(module, str) else module
            for j, a in enumerate(args):
                try:
                    args[j] = eval(a) if isinstance(a, str) else a
                except:
                    pass

            number = max(round(number * depth_multiple), 1) if number > 1 else number
            if module in [Conv2D, Conv, Bottleneck, SPP, DWConv, Focus, BottleneckCSP, C3]:
                c2 = args[0]
                c2 = math.ceil(c2 * width_multiple / 8) * 8 if c2 != output_dims else c2
                args = [c2, *args[1:]]

                if module in [BottleneckCSP, C3]:
                    args.insert(2, number)  # number of repeats
                    number = 1
            elif module is Concat:
                c2 = sum([ch[x] for x in f])
            elif module is Detect:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):  # number of anchors
                    args[1] = [list(range(args[1] * 2))] * len(f)
            else:
                c2 = ch[f]

            modules_ = tf.keras.Sequential(*[module(*args) for _ in range(number)]) if number > 1 else module(*args)
            t = str(module)[8:-2].replace('__main__.', '')  # module type
            np = sum([x.numel() for x in modules_.parameters()])  # number params
            modules_.i, modules_.f, modules_.type, modules_.np = i, f, t, np  # attach index, 'from' index, type, number params
            logger.info('%3s%18s%3s%10.0f  %-40s%-30s' % (i, f, number, np, t, args))  # print
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
            layers.append(modules_)
            if i == 0:
                ch = []
            ch.append(c2)

        return layers, sorted(save)


class Detect(Layer):
    def __init__(self, num_classes, anchors=()):
        super(Detect, self).__init__()
        self.num_classes = num_classes
        self.num_scale = len(anchors)
        self.output_dims = self.num_classes + 5
        self.num_anchors = len(anchors[0]) // 2
        self.stride = np.array([8, 16, 32], np.float32)
        self.anchors = tf.cast(tf.reshape(anchors, [self.num_anchors, -1, 2]), tf.float32)
        self.modules = [Conv2D(self.output_dims * self.num_anchors, 1, use_bias=False) for _ in range(self.num_scale)]

    def call(self, x, training=True):
        res = []
        for i in range(self.num_scale):  # number of scale layer, default=3
            y = self.modules[i](x[i])
            _, grid1, grid2, _ = y.shape
            y = tf.reshape(y, (-1, grid1, grid2, self.num_scale, self.output_dims))

            grid_xy = tf.meshgrid(tf.range(grid1), tf.range(grid2))  # grid[x][y]==(y,x)
            grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2), tf.float32)

            y_norm = tf.sigmoid(y)  # sigmoid for all dims
            xy, wh, conf, classes = tf.split(y_norm, (2, 2, 1, self.num_classes), axis=-1)

            pred_xy = (xy * 2. - 0.5 + grid_xy) * self.stride[i]  # decode pred to xywh
            pred_wh = (wh * 2) ** 2 * self.anchors[i] * self.stride[i]

            out = tf.concat([pred_xy, pred_wh, conf, classes], axis=-1)
            res.append(out)
        return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='odet.yaml', help='model.yaml')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    set_logging()
    device = select_device(opt.device)

    # Create model
    model = Network(opt.cfg)
    model.train()