# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : data_augment.py
# Description : data augmentation
# --------------------------------------
import cv2
import random
import numpy as np

def random_horizontal_flip(image, bboxes):
    """
    Randomly horizontal flip the image and correct the box
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        _, w, _ = image.shape
        image = image[:, ::-1, :]
        bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

    return image, bboxes

def random_vertical_flip(image, bboxes):
    """
    Randomly vertical flip the image and correct the box
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        h, _, _ = image.shape
        image = image[::-1, :, :]
        bboxes[:, [1, 3]] = h - bboxes[:, [3, 1]]

    return image, bboxes

def random_expand(image, bboxes, max_ratio=3, fill=0, keep_ratio=True):
    """
    Random expand original image with borders, this is identical to placing
    the original image on a larger canvas.
    :param image: BGR image data shape is [height, width, channel]
    :param bboxes: bounding box shape is [num, 4]
    :param max_ratio: Maximum ratio of the output image on both direction(vertical and horizontal)
    :param fill: The value(s) for padded borders.
    :param keep_ratio: If `True`, will keep output image the same aspect ratio as input.
    :return: result
    """
    h, w, c = image.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)

    oh, ow = int(h * ratio_y), int(w * ratio_x)
    off_y = random.randint(0, oh - h)
    off_x = random.randint(0, ow - w)

    dst = np.full(shape=(oh, ow, c), fill_value=fill, dtype=image.dtype)

    dst[off_y:off_y + h, off_x:off_x + w, :] = image

    # correct bbox
    bboxes[:, :2] += (off_x, off_y)
    bboxes[:, 2:4] += (off_x, off_y)

    return dst, bboxes

def letterbox_resize(image, target_sizes, interp=0, label=None):
    if not isinstance(target_sizes, (list, set, tuple)):
        target_sizes = [target_sizes, target_sizes]
    target_h, target_w = target_sizes

    h, w, _ = image.shape
    scale = min(target_h / h, target_w / w)
    temp_h, temp_w = int(scale * h), int(scale * w)

    image_resize = cv2.resize(image, (temp_w, temp_h), interpolation=interp)
    image_padded = np.full(shape=(target_h, target_w, 3), fill_value=128.0)

    delta_h, delta_w = (target_h - temp_h) // 2, (target_w - temp_w) // 2
    image_padded[delta_h: delta_h + temp_h, delta_w: delta_w + temp_w, :] = image_resize

    if label is not None:
        label[:, [0, 2]] = (label[:, [0, 2]] * scale * w + delta_w) / target_w
        label[:, [1, 3]] = (label[:, [1, 3]] * scale * h + delta_h) / target_h
        return image_padded, label
    else:
        return image_padded

def random_color_distort(image, brightness=32, hue=18, saturation=0.5, value=0.5):
    """
    randomly distort image color include brightness, hue, saturation, value.
    :param image: BGR image data shape is [height, width, channel]
    :param brightness:
    :param hue:
    :param saturation:
    :param value:
    :return: result
    """
    def random_hue(image_hsv, hue):
        if random.random() < 0.5:
            hue_delta = np.random.randint(-hue, hue)
            image_hsv[:, :, 0] = (image_hsv[:, :, 0] + hue_delta) % 180
        return image_hsv

    def random_saturation(image_hsv, saturation):
        if random.random() < 0.5:
            saturation_mult = 1 + np.random.uniform(-saturation, saturation)
            image_hsv[:, :, 1] *= saturation_mult
        return image_hsv

    def random_value(image_hsv, value):
        if random.random() < 0.5:
            value_mult = 1 + np.random.uniform(-value, value)
            image_hsv[:, :, 2] *= value_mult
        return image_hsv

    def random_brightness(image, brightness):
        if random.random() < 0.5:
            image = image.astype(np.float32)
            brightness_delta = int(np.random.uniform(-brightness, brightness))
            image = image + brightness_delta
        return np.clip(image, 0, 255)

    # brightness
    image = random_brightness(image, brightness)
    image = image.astype(np.uint8)

    # color jitter
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    if np.random.randint(0, 2):
        image_hsv = random_value(image_hsv, value)
        image_hsv = random_saturation(image_hsv, saturation)
        image_hsv = random_hue(image_hsv, hue)
    else:
        image_hsv = random_saturation(image_hsv, saturation)
        image_hsv = random_hue(image_hsv, hue)
        image_hsv = random_value(image_hsv, value)

    image_hsv = np.clip(image_hsv, 0, 255)
    image = cv2.cvtColor(image_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return image

def mix_up(image_1, image_2, bbox_1, bbox_2):
    """
    Overlay images and tags
    :param image_1: BGR image_1 data shape is [height, width, channel]
    :param image_2: BGR image_2 data shape is [height, width, channel]
    :param bbox_1: bounding box_1 shape is [num, 4]
    :param bbox_2: bounding box_2 shape is [num, 4]
    :return:
    """
    height = max(image_1.shape[0], image_2.shape[0])
    width = max(image_1.shape[1], image_2.shape[1])

    mix_image = np.zeros(shape=(height, width, 3), dtype='float32')

    rand_num = np.random.beta(1.5, 1.5)
    rand_num = max(0, min(1, rand_num))

    mix_image[:image_1.shape[0], :image_1.shape[1], :] = image_1.astype('float32') * rand_num
    mix_image[:image_2.shape[0], :image_2.shape[1], :] += image_2.astype('float32') * (1. - rand_num)

    mix_image = mix_image.astype('uint8')

    # the last element of the 2nd dimention is the mix up weight
    bbox_1 = np.concatenate((bbox_1, np.full(shape=(bbox_1.shape[0], 1), fill_value=rand_num)), axis=-1)
    bbox_2 = np.concatenate((bbox_2, np.full(shape=(bbox_2.shape[0], 1), fill_value=1. - rand_num)), axis=-1)
    mix_bbox = np.concatenate((bbox_1, bbox_2), axis=0)
    mix_bbox = mix_bbox.astype(np.int32)

    return mix_image, mix_bbox

def random_crop(image, bboxes):
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
        crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
        crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
        crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

        image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

    return image, bboxes

def random_translate(image, bboxes):
    """
    translation image and bboxes
    :param image: BGR image data shape is [height, width, channel]
    :param bbox: bounding box_1 shape is [num, 4]
    :return: result
    """
    if random.random() < 0.5:
        h, w, _ = image.shape
        max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

        max_l_trans = max_bbox[0]
        max_u_trans = max_bbox[1]
        max_r_trans = w - max_bbox[2]
        max_d_trans = h - max_bbox[3]

        tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
        ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

        M = np.array([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h))

        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
        bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

    return image, bboxes