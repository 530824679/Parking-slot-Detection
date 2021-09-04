#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2021/5/21
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : config.py
# @Description : set global config
# @IDE: PyCharm Community Edition
# --------------------------------------

import os

path_params = {
    'data_path': "/home/chenwei/HDD/Project/Parking-slot-Detection/datasets/voc",
    'class_file': '../data/classes.txt',
    'train_file': '/home/chenwei/HDD/Project/Parking-slot-Detection/data/train.txt',
    'anchor_file': '/home/chenwei/HDD/Project/Parking-slot-Detection/data/anchors.txt',
    'tfrecord_dir': '/home/chenwei/HDD/Project/Parking-slot-Detection/tfrecords',
    'logs_dir': './logs',
    'ckpt_name': 'Fisheye_OD',
    'ckpt_dir': './checkpoints',
    'initial_weight': './weight/model.ckpt'
}

data_params = {
    'angle': 0,
    'saturation': 1.5,
    'exposure': 1.5,
    'hue': .1,
    'jitter': .3,
}

model_params = {
    'input_height': 416,
    'input_width': 416,
    'channel': 3,
    'classes': 3,
    'depth': 0.33,       # [0.33, 0.67, 1.0, 1.33]
    'width': 0.50,       # [0.50, 0.75, 1.0, 1.25]
    'strides': [8, 16, 32],
    'anchor_per_scale': 3,
    'label_smoothing': 0.01,
    'iou_threshold': 0.3,
    'warm_up_epoch': 3,
    'anchors': [8., 9., 16., 24., 28., 58., 41., 25., 58., 125., 71., 52., 129., 97., 163., 218., 384.,347.]
}

solver_params = {
    'total_epoches': 200,
    'batch_size': 8,
    'init_learning_rate': 3e-4,
    'warmup_learning_rate': 1e-6,
    'warmup_epoches': 2,
    'decay_steps': 500,             # 衰变步数
    'decay_rate': 0.95,             # 衰变率
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'restore': False,
}



