

# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : loss.py
# Description :Yolo_v5 Loss损失函数.
# --------------------------------------

import math
import numpy as np
import tensorflow as tf
from cfg.config import *
from utils.data_utils import *

class Loss(object):
    def __init__(self):
        self.batch_size = solver_params['batch_size']
        self.anchors = read_anchors(path_params['anchor_file'])
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.classes = read_class_names(path_params['class_file'])
        self.class_num = len(self.classes)
        self.label_smoothing = model_params['label_smoothing']
        self.iou_threshold = model_params['iou_threshold']

