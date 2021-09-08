import cv2
import numpy as np
import tensorflow as tf
from utils.tf_utils import xyxy2xywh
from data.data_augment import letterbox_resize
from data.label_anchor import AnchorLabeler

class DataLoader(object):
    def __init__(self, data_reader, anchors, stride, img_size=416, anchor_assign_method='wh', anchor_positive_augment=True):
        self.data_reader = data_reader
        self.anchor_label = AnchorLabeler(anchors,
                                          grids=img_size / stride,
                                          img_size=img_size,
                                          assign_method=anchor_assign_method,
                                          extend_offset=anchor_positive_augment)
        self.img_size = img_size

    def __call__(self, batch_size=8, anchor_label=True):
        dataset = tf.data.Dataset.from_generator(self.data_reader.iter,
                                                 output_types=(tf.float32, tf.float32),
                                                 output_shapes=([self.img_size, self.img_size, 3], [None, 5]))

        if anchor_label:  # when train
            dataset = dataset.map(self.transform, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    def transform(self, image, label):
        label_encoder = self.anchor_label.encode(label)
        return image, label_encoder

class DataReader(object):
    def __init__(self, train_list, img_size=640, transforms=None, mosaic=False, augment=False, filter_idx=None):
        self.train_list = train_list
        self.labels = self.load_labels(train_list)
        self.idx = range(len(self.labels))
        self.img_size = img_size
        self.transforms = transforms
        self.mosaic = mosaic
        self.augment = augment
        self.images_dir = []
        self.labels_ori = []

        if filter_idx is not None:
            self.idx = [i for i in self.idx if i in filter_idx]
            print('filter {} from {}'.format(len(self.idx), len(self.labels)))

        for i in self.idx:
            image_dir, label = self.parse_labels(self.labels[i])
            self.images_dir.append(image_dir)
            self.labels_ori.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image, label = self.load_image_and_label(idx)

        if self.transforms:
            image, label = self.transforms(image, label, mosaic=self.mosaic, augment=self.augment)
        image, label = letterbox_resize(image, self.img_size, label=label)
        label[:, 0:4] = xyxy2xywh(label[:, 0:4])
        return image, label

    def iter(self):
        for i in self.idx:
            yield self[i]

    def load_labels(self, label_list):
        with open(label_list, 'r') as f:
            labels = [line.strip() for line in f.readlines() if len(line.strip().split()[1:]) != 0]
        print('Load examples : {}'.format(len(labels)))
        if 'train' in label_list:
            np.random.shuffle(labels)
        return labels

    def parse_labels(self, labels):
        example = labels.split()
        image_dir = example[0]

        label = np.array([list(map(float, box.split(',')[0: 5])) for box in example[1:]])
        return image_dir, label

    def load_image(self, idx):
        img_dir = self.images_dir[idx]
        image = cv2.imread(img_dir)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        return image

    def load_image_and_label(self, idx):
        image = self.load_image(idx)
        label = self.labels_ori[idx].copy()
        return image, label

    def transforms(self, image, label, mosaic, augment):
        image = image / 255.
        if np.max(label[:, 0:4]) > 1:
            label[:, [0, 2]] = label[:, [0, 2]] / image.shape[1]
            label[:, [1, 3]] = label[:, [1, 3]] / image.shape[0]
        return image, label