# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : metrics.py
# Description :Loss function definition
# --------------------------------------

import math
import numpy as np
import tensorflow as tf

class Loss(object):
    def __init__(self, anchors, num_classes, img_size, label_smoothing=0):
        self.anchors = anchors
        self.strides = [8, 16, 32]
        self.num_classes = num_classes
        self.img_size = img_size
        self.bce_conf = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.bce_class = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE, label_smoothing=label_smoothing)

    def __call__(self, y_true, y_pred):
        iou_loss_all = obj_loss_all = class_loss_all = 0
        balance = [1.0, 1.0, 1.0] if len(y_pred) == 3 else [4.0, 1.0, 0.4, 0.1]  # P3-5 or P3-6

        for i, (pred, true) in enumerate(zip(y_pred, y_true)):
            # preprocess, true: batch_size * grid * grid * 3 * 6, pred: batch_size * grid * grid * clss+5
            true_box, true_obj, true_class = tf.split(true, (4, 1, -1), axis=-1)
            pred_box, pred_obj, pred_class = tf.split(pred, (4, 1, -1), axis=-1)
            if tf.shape(true_class)[-1] == 1 and self.num_classes > 1:
                true_class = tf.squeeze(tf.one_hot(tf.cast(true_class, tf.dtypes.int32), depth=self.num_classes, axis=-1), -2)

            # prepare: higher weights to smaller box, true_wh should be normalized to (0,1)
            box_scale = 2 - 1.0 * true_box[..., 2] * true_box[..., 3] / (self.img_size ** 2)
            obj_mask = tf.squeeze(true_obj, -1)  # obj or noobj, batch_size * grid * grid * anchors_per_grid
            background_mask = 1.0 - obj_mask
            conf_focal = tf.squeeze(tf.math.pow(true_obj - pred_obj, 2), -1)

            # iou/ giou/ ciou/ diou loss
            iou = bbox_iou(pred_box, true_box, xyxy=False, giou=True)
            iou_loss = (1 - iou) * obj_mask * box_scale  # batch_size * grid * grid * 3

            # confidence loss, Todo: multiply the iou
            conf_loss = self.bce_conf(true_obj, pred_obj)
            conf_loss = conf_focal * (obj_mask * conf_loss + background_mask * conf_loss)  # batch * grid * grid * 3

            # class loss
            # use binary cross entropy loss for multi class, so every value is independent and sigmoid
            # please note that the output of tf.keras.losses.bce is original dim minus the last one
            class_loss = obj_mask * self.bce_class(true_class, pred_class)

            iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3]))
            conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3]))
            class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3]))

            iou_loss_all += iou_loss * balance[i]
            obj_loss_all += conf_loss * balance[i]
            class_loss_all += class_loss * self.num_classes * balance[i]  # to balance the 3 loss

        try:
            print('-' * 55, 'iou', tf.reduce_sum(iou_loss_all).numpy(), ', conf', tf.reduce_sum(obj_loss_all).numpy(), ', class', tf.reduce_sum(class_loss_all).numpy())
        except:  # tf graph mode
            pass
        return (iou_loss_all, obj_loss_all, class_loss_all)

    def focal_loss(self, target, actual, alpha=0.25, gamma=2):
        focal_loss = tf.abs(alpha + target - 1) * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def smooth_labels(self, y_true, label_smoothing=0.01):
        # smooth labels
        label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)
        uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
        smooth_onehot = y_true * (1 - label_smoothing) + label_smoothing * uniform_distribution
        return smooth_onehot

def bbox_iou(bbox1, bbox2, xyxy=False, giou=False, diou=False, ciou=False, epsilon=1e-9):
    assert bbox1.shape == bbox2.shape

    if xyxy:
        b1x1, b1y1, b1x2, b1y2 = bbox1[..., 0], bbox1[..., 1], bbox1[..., 2], bbox1[..., 3]
        b2x1, b2y1, b2x2, b2y2 = bbox2[..., 0], bbox2[..., 1], bbox2[..., 2], bbox2[..., 3]
    else:  # xywh -> xyxy
        b1x1, b1x2 = bbox1[..., 0] - bbox1[..., 2] / 2, bbox1[..., 0] + bbox1[..., 2] / 2
        b1y1, b1y2 = bbox1[..., 1] - bbox1[..., 3] / 2, bbox1[..., 1] + bbox1[..., 3] / 2
        b2x1, b2x2 = bbox2[..., 0] - bbox2[..., 2] / 2, bbox2[..., 0] + bbox2[..., 2] / 2
        b2y1, b2y2 = bbox2[..., 1] - bbox2[..., 3] / 2, bbox2[..., 1] + bbox2[..., 3] / 2

    # intersection area
    inter = tf.maximum(tf.minimum(b1x2, b2x2) - tf.maximum(b1x1, b2x1), 0) * \
            tf.maximum(tf.minimum(b1y2, b2y2) - tf.maximum(b1y1, b2y1), 0)

    # union area
    w1, h1 = b1x2 - b1x1 + epsilon, b1y2 - b1y1 + epsilon
    w2, h2 = b2x2 - b2x1 + epsilon, b2y2 - b2y1 + epsilon
    union = w1 * h1 + w2 * h2 - inter + epsilon

    # iou
    iou = inter / union

    if giou or diou or ciou:
        # enclosing box
        cw = tf.maximum(b1x2, b2x2) - tf.minimum(b1x1, b2x1)
        ch = tf.maximum(b1y2, b2y2) - tf.minimum(b1y1, b2y1)
        if giou:
            enclose_area = cw * ch + epsilon
            giou = iou - 1.0 * (enclose_area - union) / enclose_area
            return tf.clip_by_value(giou, -1, 1)
        if diou or ciou:
            c2 = cw ** 2 + ch ** 2 + epsilon
            rho2 = ((b2x1 + b2x2) - (b1x1 + b1x2)) ** 2 / 4 + ((b2y1 + b2y2) - (b1y1 + b1y2)) ** 2 / 4
            if diou:
                return iou - rho2 / c2
            elif ciou:
                v = (4 / math.pi ** 2) * tf.pow(tf.atan(w2 / h2) - tf.atan(w1 / h1), 2)
                alpha = v / (1 - iou + v)
                return iou - (rho2 / c2 + v * alpha)
    return tf.clip_by_value(iou, 0, 1)











def loss_layer(self, logits, y_true, anchors):
    feature_size = tf.shape(logits)[1:3]
    ratio = tf.cast(self.input_size / feature_size, tf.float32)

    # ground truth
    object_coords = y_true[:, :, :, :, 0:4]
    object_masks = y_true[:, :, :, :, 4:5]
    object_probs = y_true[:, :, :, :, 5:]

    # shape: [N, N, N, 5, 4] & [N, N, N, 5] ==> [V, 4]
    valid_true_boxes = tf.boolean_mask(object_coords, tf.cast(object_masks[..., 0], 'bool'))
    # shape: [V, 2]
    valid_true_box_xy = valid_true_boxes[:, 0:2]
    valid_true_box_wh = valid_true_boxes[:, 2:4]

    # predicts
    xy_cell, bboxes_xywh, pred_conf_logits, pred_prob_logits = self.reorg_layer(logits, anchors)
    pred_box_xy = bboxes_xywh[..., 0:2]
    pred_box_wh = bboxes_xywh[..., 2:4]

    box_loss_scale = 2. - (1.0 * y_true[..., 2:3] / tf.cast(self.input_width, tf.float32)) * (
                1.0 * y_true[..., 3:4] / tf.cast(self.input_height, tf.float32))
    true_xy = y_true[..., 0:2] / ratio[::-1] - xy_cell
    pred_xy = pred_box_xy / ratio[::-1] - xy_cell

    true_tw_th = y_true[..., 2:4] / anchors
    pred_tw_th = pred_box_wh / anchors
    true_tw_th = tf.where(condition=tf.equal(true_tw_th, 0), x=tf.ones_like(true_tw_th), y=true_tw_th)
    pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)
    true_tw_th = tf.log(tf.clip_by_value(true_tw_th, 1e-9, 1e9))
    pred_tw_th = tf.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))

    xy_loss = tf.square(true_xy - pred_xy) * object_masks * box_loss_scale
    wh_loss = tf.square(true_tw_th - pred_tw_th) * object_masks * box_loss_scale

    diou = tf.expand_dims(self.bbox_diou(bboxes_xywh, object_coords), axis=-1)
    iou_loss = object_masks * box_loss_scale * (1 - diou)

    # confidence loss
    iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
    best_iou = tf.reduce_max(iou, axis=-1)
    ignore_mask = tf.expand_dims(tf.cast(best_iou < self.iou_threshold, tf.float32), -1)

    conf_pos_mask = object_masks
    conf_neg_mask = (1 - object_masks) * ignore_mask
    conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks,
                                                                            logits=pred_conf_logits)
    conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks,
                                                                            logits=pred_conf_logits)
    conf_loss = conf_loss_pos + conf_loss_neg

    # focal_loss
    alpha = 1.0
    gamma = 2.0
    focal_mask = alpha * tf.pow(tf.abs(object_masks - tf.sigmoid(pred_conf_logits)), gamma)
    conf_loss = conf_loss * focal_mask

    # class loss
    delta = 0.01
    label_target = (1 - delta) * object_probs + delta * 1. / self.class_num
    class_loss = object_masks * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=pred_prob_logits)

    iou_loss = tf.reduce_mean(tf.reduce_sum(iou_loss, axis=[1, 2, 3, 4]))
    # xy_loss = tf.reduce_mean(tf.reduce_sum(xy_loss, axis=[1, 2, 3, 4]))
    # wh_loss = tf.reduce_mean(tf.reduce_sum(wh_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    class_loss = tf.reduce_mean(tf.reduce_sum(class_loss, axis=[1, 2, 3, 4]))
    return iou_loss, conf_loss, class_loss


def reorg_layer(self, feature_maps, anchors):
    """
    解码网络输出的特征图
    :param feature_maps:网络输出的特征图
    :param anchors:当前层使用的anchor尺度
    :param stride:特征图相比原图的缩放比例
    :return: 预测层最终的输出 shape=[batch_size, feature_size, feature_size, anchor_per_scale, 5 + class_num]
    """
    feature_shape = feature_maps.get_shape().as_list()[1:3]
    ratio = tf.cast(self.input_size / feature_shape, tf.float32)
    anchor_per_scale = self.anchor_per_sacle

    # rescale the anchors to the feature map [w, h]
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    # 网络输出转化——偏移量、置信度、类别概率
    feature_maps = tf.reshape(feature_maps,
                              [tf.shape(feature_maps)[0], feature_shape[0], feature_shape[1], anchor_per_scale,
                               self.class_num + 5])
    # 中心坐标相对于该cell左上角的偏移量，sigmoid函数归一化到0-1
    xy_offset = tf.nn.sigmoid(feature_maps[:, :, :, :, 0:2])
    # 相对于anchor的wh比例，通过e指数解码
    wh_offset = tf.clip_by_value(tf.exp(feature_maps[:, :, :, :, 2:4]), 1e-9, 50)
    # 置信度，sigmoid函数归一化到0-1
    conf_logits = feature_maps[:, :, :, :, 4:5]
    # 网络回归的是得分,用softmax转变成类别概率
    prob_logits = feature_maps[:, :, :, :, 5:]

    # 构建特征图每个cell的左上角的xy坐标
    height_index = tf.range(feature_shape[0], dtype=tf.int32)
    width_index = tf.range(feature_shape[1], dtype=tf.int32)
    x_cell, y_cell = tf.meshgrid(width_index, height_index)

    x_cell = tf.reshape(x_cell, (-1, 1))
    y_cell = tf.reshape(y_cell, (-1, 1))
    xy_cell = tf.concat([x_cell, y_cell], axis=-1)
    xy_cell = tf.cast(tf.reshape(xy_cell, [feature_shape[0], feature_shape[1], 1, 2]), tf.float32)

    bboxes_xy = (xy_cell + xy_offset) * ratio[::-1]
    bboxes_wh = (rescaled_anchors * wh_offset) * ratio[::-1]

    bboxes_xywh = tf.concat([bboxes_xy, bboxes_wh], axis=-1)

    return xy_cell, bboxes_xywh, conf_logits, prob_logits


def broadcast_iou(self, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
    # shape:
    # true_box_??: [V, 2] V:目标数量
    # pred_box_??: [N, 13, 13, 3, 2]

    # shape: [N, 13, 13, 3, 1, 2] , 扩张维度方便进行维度广播
    pred_box_xy = tf.expand_dims(pred_box_xy, -2)
    pred_box_wh = tf.expand_dims(pred_box_wh, -2)

    # shape: [1, V, 2] V:该尺度下分feature_map 下所有的目标是目标数量
    true_box_xy = tf.expand_dims(true_box_xy, 0)
    true_box_wh = tf.expand_dims(true_box_wh, 0)

    # [N, 13, 13, 3, 1, 2] --> [N, 13, 13, 3, V, 2] & [1, V, 2] ==> [N, 13, 13, 3, V, 2] 维度广播
    # 真boxe,左上角,右下角, 假boxe的左上角,右小角,
    intersect_mins = tf.maximum(pred_box_xy - pred_box_wh / + 2.,  # 取最靠右的左上角
                                true_box_xy - true_box_wh / 2.)
    intersect_maxs = tf.minimum(pred_box_xy + pred_box_wh / 2.,  # 取最靠左的右下角
                                true_box_xy + true_box_wh / 2.)
    # tf.maximun 去除那些没有面积交叉的矩形框, 置0
    intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)  # 得到重合区域的长和宽

    # shape: [N, 13, 13, 3, V]
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]  # 重合部分面积
    # shape: [N, 13, 13, 3, 1]
    pred_box_area = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # 预测区域面积
    # shape: [1, V]
    true_box_area = true_box_wh[..., 0] * true_box_wh[..., 1]  # 真实区域面积
    # [N, 13, 13, 3, V]
    iou = intersect_area / (pred_box_area + true_box_area - intersect_area + 1e-10)

    return iou


