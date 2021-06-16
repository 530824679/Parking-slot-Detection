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
    'class_file': '/home/chenwei/HDD/Project/Parking-slot-Detection/data/classes.txt',
    'train_file': '/home/chenwei/HDD/Project/Parking-slot-Detection/data/train.txt',
    'anchor_file': '/home/chenwei/HDD/Project/Parking-slot-Detection/data/anchors.txt',
    'tfrecord_dir': '/home/chenwei/HDD/Project/Parking-slot-Detection/tfrecords',
    'logs_dir': './logs',
    'checkpoint_name': 'Fisheye_OD',
    'checkpoints_dir': './checkpoints',
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
    'input_height': 480,
    'input_width': 640,
    'channel': 3,
    'classes': 3,
    'strides': [8, 16, 32],
    'anchor_per_scale': 3,
    'label_smoothing': 0.01,
    'iou_threshold': 0.5,
    'warm_up_epoch': 3,
    'anchors': [27,22, 20,36, 42,32, 34,56, 63,49, 60,94, 104,75, 142,139, 290,246]
}

solver_params = {
    'total_epoches': 20000,
    'batch_size': 4,
    'warmup_epoches': 10,
    'learning_rate': 0.0001,
    'decay_steps': 500,            # 衰变步数
    'decay_rate': 0.95,             # 衰变率
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'restore': False,
}



