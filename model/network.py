# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : network.py
# Description :YOLO v5 network architecture
# --------------------------------------

import math
import numpy as np
import tensorflow as tf
from cfg.config import *
from model.basenet import *
from utils.data_utils import *

class Network(object):
    def __init__(self, is_train):
        self.is_train = is_train
        self.input_height = model_params['input_height']
        self.input_width = model_params['input_width']
        self.input_size = np.array([self.input_height, self.input_width])
        self.anchors = read_anchors(path_params['anchor_file'])
        self.anchor_per_sacle = model_params['anchor_per_scale']
        self.classes = read_class_names(path_params['class_file'])
        self.class_num = len(self.classes)
        self.strides = np.array(model_params['strides'])
        self.anchor_per_scale = model_params['anchor_per_scale']
        self.label_smoothing = model_params['label_smoothing']
        self.iou_threshold = model_params['iou_threshold']

    def forward(self, inputs):
        with tf.variable_scope('CSPnet'):
            try:
                conv_lbbox, conv_mbbox, conv_sbbox = self.build_network(inputs)
            except:
                raise NotImplementedError("Can not build up yolov5 network!")

            logits = [conv_sbbox, conv_mbbox, conv_lbbox]

            return logits

    def build_network(self, inputs):

        # backbone
        with tf.variable_scope('backbone'):
            focus_0 = focus(inputs, 64, 3, 'model/0')
            conv_1 = convBnLeakly(focus_0, 128, 3, 2, "model/1")
            bottleneck_csp_2 = bottleneckCSP(conv_1, 128, 128, 3, True, 0.5, "model/2")
            conv_3 = convBnLeakly(bottleneck_csp_2, 256, 3, 2, 'model/3')
            bottleneck_csp_4 = bottleneckCSP(conv_3, 256, 256, 9, True, 0.5, 'model/4')
            conv_5 = convBnLeakly(bottleneck_csp_4, 512, 3, 2, 'model/5')
            bottleneck_csp_6 = bottleneckCSP(conv_5, 512, 512, 9, True, 0.5, 'model/6')
            conv_7 = convBnLeakly(bottleneck_csp_6, 1024, 3, 2, 'model/7')
            spp_8 = spp(conv_7, 1024, 1024, 5, 9, 13, 'model/8')

        # neck
        with tf.variable_scope('neck'):
            bottleneck_csp_9 = bottleneckCSP(spp_8, 1024, 1024, 3, False, 0.5, 'model/9')
            conv_10 = convBnLeakly(bottleneck_csp_9, 512, 1, 1, 'model/10')

            shape = [conv_10.shape[1].value * 2, conv_10.shape[2].value * 2]
            # 0：双线性差值。1：最近邻居法。2：双三次插值法。3：面积插值法
            deconv_11 = tf.image.resize_images(conv_10, shape, method=1)

            concat_12 = tf.concat((deconv_11, bottleneck_csp_6), -1)
            bottleneck_csp_13 = bottleneckCSP(concat_12, 1024, 512, 3, False, 0.5, 'model/13')
            conv_14 = convBnLeakly(bottleneck_csp_13, 256, 1, 1, 'model/14')

            shape = [conv_14.shape[1].value * 2, conv_14.shape[2].value * 2]
            deconv_15 = tf.image.resize_images(conv_14, shape, method=1)

            concat_16 = tf.concat((deconv_15, bottleneck_csp_4), -1)
            bottleneck_csp_17 = bottleneckCSP(concat_16, 512, 256, 3, False, 0.5, 'model/17')
            conv_18 = convBnLeakly(bottleneck_csp_17, 256, 3, 2, 'model/18')

            concat_19 = tf.concat((conv_18, conv_14), -1)
            bottleneck_csp_20 = bottleneckCSP(concat_19, 512, 512, 3, False, 0.5, 'model/20')
            conv_21 = convBnLeakly(bottleneck_csp_20, 512, 3, 2, 'model/21')

            concat_22 = tf.concat((conv_21, conv_10), -1)
            bottleneck_csp_23 = bottleneckCSP(concat_22, 1024, 1024, 3, False, 0.5, 'model/23')

        # head
        with tf.variable_scope('head'):
            conv_24_m0 = conv(bottleneck_csp_17, 3 * (self.class_num + 5), 1, 1, 'model/24/m/0', add_bias=True)
            conv_24_m1 = conv(bottleneck_csp_20, 3 * (self.class_num + 5), 1, 1, 'model/24/m/1', add_bias=True)
            conv_24_m2 = conv(bottleneck_csp_23, 3 * (self.class_num + 5), 1, 1, 'model/24/m/2', add_bias=True)

        return conv_24_m0, conv_24_m1, conv_24_m2

    def calc_loss(self, y_logit, y_true):
        """
        :param pred_conv: [pred_sconv, pred_mconv, pred_lconv]. pred_conv_shape=[batch_size, conv_height, conv_width, anchor_per_scale, 5 + num_classes]
        :param label_bbox: [label_sbbox, label_mbbox, label_lbbox].
        :return:
        """
        loss_iou, loss_conf, loss_class = 0., 0., 0.
        anchor_group = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]

        for i in range(len(y_logit)):
            result = self.loss_layer(y_logit[i], y_true[i], anchor_group[i])
            loss_iou += result[0]
            loss_conf += result[1]
            loss_class += result[2]
        total_loss = loss_iou + loss_conf + loss_class
        return [total_loss, loss_iou, loss_conf, loss_class]

    def loss_layer(self, logits, y_true, anchors):

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


        box_loss_scale = 2. - (1.0 * y_true[..., 2:3] / tf.cast(self.input_width, tf.float32)) * (1.0 * y_true[..., 3:4] / tf.cast(self.input_height, tf.float32))
        ciou = tf.expand_dims(self.box_ciou(bboxes_xywh, object_coords), axis=-1)
        iou_loss = object_masks * box_loss_scale * (1 - ciou)

        # confidence loss
        iou = self.broadcast_iou(valid_true_box_xy, valid_true_box_wh, pred_box_xy, pred_box_wh)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask = tf.expand_dims(tf.cast(best_iou < self.iou_threshold, tf.float32), -1)

        conf_pos_mask = object_masks
        conf_neg_mask = (1 - object_masks) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_masks, logits=pred_conf_logits)
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
        feature_maps = tf.reshape(feature_maps, [tf.shape(feature_maps)[0], feature_shape[0], feature_shape[1], anchor_per_scale, self.class_num + 5])
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

    def focal_loss(self, target, actual, alpha=0.25, gamma=2):
        focal_loss = tf.abs(alpha + target - 1) * tf.pow(tf.abs(target - actual), gamma)
        return focal_loss

    def smooth_labels(self, y_true, label_smoothing=0.01):
        # smooth labels
        label_smoothing = tf.constant(label_smoothing, dtype=tf.float32)
        uniform_distribution = np.full(self.class_num, 1.0 / self.class_num)
        smooth_onehot = y_true * (1 - label_smoothing) + label_smoothing * uniform_distribution
        return smooth_onehot

    def bbox_iou(self, boxes_1, boxes_2):
        """
        calculate regression loss using iou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        return iou

    def bbox_giou(self, boxes_1, boxes_2):
        """
        calculate regression loss using giou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate iou add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate area of the minimun closed convex surface
        enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]

        # calculate the giou add epsilon in denominator to avoid dividing by 0
        giou = iou - 1.0 * (enclose_area - union_area) / (enclose_area + tf.keras.backend.epsilon())

        return giou

    def bbox_diou(self, boxes_1, boxes_2):
        """
        calculate regression loss using diou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou add epsilon in denominator to avoid dividing by 0
        diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

        return diou

    def box_ciou(self, boxes_1, boxes_2):
        """
        calculate regression loss using ciou
        :param boxes_1: boxes_1 shape is [x, y, w, h]
        :param boxes_2: boxes_2 shape is [x, y, w, h]
        :return:
        """
        # calculate center distance
        center_distance = tf.reduce_sum(tf.square(boxes_1[..., :2] - boxes_2[..., :2]), axis=-1)

        v = 4 * tf.square(tf.math.atan2(boxes_1[..., 2], boxes_1[..., 3]) - tf.math.atan2(boxes_2[..., 2], boxes_2[..., 3])) / (math.pi * math.pi)

        # transform [x, y, w, h] to [x_min, y_min, x_max, y_max]
        boxes_1 = tf.concat([boxes_1[..., :2] - boxes_1[..., 2:] * 0.5,
                             boxes_1[..., :2] + boxes_1[..., 2:] * 0.5], axis=-1)
        boxes_2 = tf.concat([boxes_2[..., :2] - boxes_2[..., 2:] * 0.5,
                             boxes_2[..., :2] + boxes_2[..., 2:] * 0.5], axis=-1)
        boxes_1 = tf.concat([tf.minimum(boxes_1[..., :2], boxes_1[..., 2:]),
                             tf.maximum(boxes_1[..., :2], boxes_1[..., 2:])], axis=-1)
        boxes_2 = tf.concat([tf.minimum(boxes_2[..., :2], boxes_2[..., 2:]),
                             tf.maximum(boxes_2[..., :2], boxes_2[..., 2:])], axis=-1)

        # calculate area of boxes_1 boxes_2
        boxes_1_area = (boxes_1[..., 2] - boxes_1[..., 0]) * (boxes_1[..., 3] - boxes_1[..., 1])
        boxes_2_area = (boxes_2[..., 2] - boxes_2[..., 0]) * (boxes_2[..., 3] - boxes_2[..., 1])

        # calculate the two corners of the intersection
        left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
        right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate area of intersection
        inter_section = tf.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]

        # calculate union area
        union_area = boxes_1_area + boxes_2_area - inter_area

        # calculate IoU, add epsilon in denominator to avoid dividing by 0
        iou = inter_area / (union_area + tf.keras.backend.epsilon())

        # calculate the upper left and lower right corners of the minimum closed convex surface
        enclose_left_up = tf.minimum(boxes_1[..., :2], boxes_2[..., :2])
        enclose_right_down = tf.maximum(boxes_1[..., 2:], boxes_2[..., 2:])

        # calculate width and height of the minimun closed convex surface
        enclose_wh = tf.maximum(enclose_right_down - enclose_left_up, 0.0)

        # calculate enclosed diagonal distance
        enclose_diagonal = tf.reduce_sum(tf.square(enclose_wh), axis=-1)

        # calculate diou
        diou = iou - 1.0 * center_distance / (enclose_diagonal + tf.keras.backend.epsilon())

        # calculate param v and alpha to CIoU
        alpha = v / (1.0 - iou + v)

        # calculate ciou
        ciou = diou - alpha * v

        return ciou