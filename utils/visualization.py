# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : visualization.py
# Description :visualization file.
# --------------------------------------

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import os
import cv2

def plot_one_box(image, coord, label=None, color=None, line_thickness=None):
    '''
    coord: [x_min, y_min, x_max, y_max] format coordinates.
    image: image to plot on.
    label: str. The label name.
    color: int. color index.
    line_thickness: int. rectangle line thickness.
    '''
    tl = line_thickness or int(round(0.002 * max(image.shape[0:2])))  # line thickness
    c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))

    cv2.rectangle(image, c1, c2, color)#, thickness=2
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]

        x1, y1 = c1[0], c1[1]
        y1 = y1 if y1 > 20 else y1 + 20
        x2 , y2 = x1 + t_size[0], y1 - t_size[1]

        cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)  # filled
        cv2.putText(image, label, (x1, y1), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)