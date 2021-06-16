
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2

import numpy as np
import tensorflow as tf
from cfg.config import *

# def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     # Resize image to a 32-pixel-multiple rectangle
#     shape = image.shape[:2]    # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#
#     return image, ratio, (dw, dh)

def post_process(inputs, grids, strides, anchors, class_num):

    total = []
    for i, logits in enumerate(inputs):

        logits_xy = (logits[..., :2] * 2. - 0.5 + grids[i]) * strides[i]
        logits_wh = ((logits[..., 2:4] * 2) ** 2) * anchors[i]
        logits_new = tf.concat((logits_xy, logits_wh, logits[..., 4:]), axis=-1)

    # 过滤低置信度的目标
    mask = total[:, 4] > 0.15
    total = tf.boolean_mask(total, mask)

    # x,y,w,h ——> x1,y1,x2,y2
    x, y, w, h, conf, prob = tf.split(total, [1, 1, 1, 1, 1, class_num], axis=-1)
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x1 + w
    y2 = y1 + h

    conf_prob = conf * prob
    scores = tf.reduce_max(conf_prob, axis=-1)
    labels = tf.cast(tf.argmax(conf_prob, axis=-1), tf.float32)
    boxes = tf.concat([x1, y1, x2, y2], axis=1)

    indices = tf.image.non_max_suppression(boxes, scores, max_output_size=1000, iou_threshold=test_params['iou_threshold'], score_threshold=test_params['score_threshold'])

    boxes = tf.gather(boxes, indices)
    scores = tf.reshape(tf.gather(scores, indices), [-1, 1])
    labels = tf.reshape(tf.gather(labels, indices), [-1, 1])

    output = tf.concat([boxes, scores, labels], axis=-1)
    return output
