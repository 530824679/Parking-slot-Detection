# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : test.py
# Description :test file.
# --------------------------------------
import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)

import numpy as np
import tensorflow as tf
from cfg.config import *
from visualize.visualization import *

def load_image(image_path):
    image = cv2.imdecode(np.fromfile(image_path, np.uint8), -1)[:, :, :3]
    image = cv2.resize(image, tuple(model_params['input_shape'][::-1]))
    normalized_image = np.expand_dims(image[:, :, ::-1], 0).astype(np.float32) / 255.
    return image, normalized_image

def test_ckpt(image_path, weights_path):
    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)



def test_pb(image_path, weights_path):
    assert os.path.exists(image_path), '{:s} not exist'.format(image_path)

    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef
        with open(weights_path, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tf.import_graph_def(output_graph_def, name="")

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        input = sess.graph.get_tensor_by_name("input:0")
        output = sess.graph.get_tensor_by_name("output:0")

        image, normalized_image = load_image(image_path)

        predict = sess.run(output, feed_dict={input: normalized_image})

        for i in range(predict.shape[0]):
            box = predict[i, :4].astype(np.int32)
            score = predict[i, 4]
            label = int(predict[i, 5])
            plot_one_box(image, box, '%d_%.4f' % (label, score), (0, 255, 0))

        cv2.imshow("result", image)
        cv2.waitKey(0)


if __name__ == "__main__":
    image_dir = '/home/chenwei/HDD/Project/Parking-slot-Detection/test'
    weights_path = '/home/chenwei/HDD/Project/Parking-slot-Detection/checkpoints/odet_train_mloss=30.6506.ckpt-258'
    test_pb(image_dir, weights_path)