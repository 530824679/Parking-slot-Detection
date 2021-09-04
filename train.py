# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2020/11/01
# @Author  : Oscar Chen
# @Email   : 530824679@qq.com
# @File    : train.py
# Description : train code
# --------------------------------------

import os
import math
import shutil
import numpy as np
import tensorflow as tf
import glog as log

from cfg.config import path_params, model_params, solver_params
from model.network import Network
from data import dataset, tfrecord
from utils.data_utils import *

def config_warmup_lr(global_step, warmup_steps, name):
    with tf.variable_scope(name_or_scope=name):
        warmup_init_learning_rate = solver_params['init_learning_rate'] / 1000.0
        factor = tf.math.pow(solver_params['init_learning_rate'] / warmup_init_learning_rate, 1.0 / warmup_steps)
        warmup_lr = warmup_init_learning_rate * tf.math.pow(factor, global_step)
    return warmup_lr

def config_cosine_lr(steps, batch_size, num_epochs, name):
    with tf.variable_scope(name_or_scope=name):
        lr_init = 0.008 * batch_size / 64
        warmup_init = 0.0008
        warmup_step = steps
        decay_steps = tf.cast((num_epochs - 1) * warmup_step, tf.float32)

        linear_warmup = tf.cast(steps, dtype=tf.float32) / warmup_step * (lr_init - warmup_init)
        cosine_lr = 0.5 * lr_init * (1 + tf.cos(math.pi * tf.cast(steps, tf.float32) / decay_steps))
        lr = tf.where(steps < warmup_step, warmup_init + linear_warmup, cosine_lr)

    return lr

def config_optimizer(optimizer_name, lr_init, momentum=0.99):
    log.info("message:配置优化器:'" + str(optimizer_name) + "'")
    if optimizer_name == 'momentum':
        return tf.train.MomentumOptimizer(learning_rate=lr_init, momentum=momentum)
    elif optimizer_name == 'adam':
        return tf.train.AdamOptimizer(learning_rate=lr_init)
    elif optimizer_name == 'sgd':
        return tf.train.GradientDescentOptimizer(learning_rate=lr_init)
    else:
        log.error("error:不支持的优化器类型:'" + str(optimizer_name) + "'")
        raise ValueError(str(optimizer_name) + ":不支持的优化器类型")

def compute_curr_epoch(global_step, batch_size, image_num):
    epoch = global_step * batch_size / image_num
    return tf.cast(epoch, tf.int32)

def check_directory():
    ckpt_path = path_params['ckpt_dir']
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    logs_path = path_params['logs_dir']
    if os.path.exists(logs_path):
        shutil.rmtree(logs_path)
    os.makedirs(logs_path)

def train():
    start_step = 0
    input_height = model_params['input_height']
    input_width = model_params['input_width']
    num_epochs = solver_params['total_epoches']
    batch_size = solver_params['batch_size']
    checkpoint_dir = path_params['checkpoints_dir']
    tfrecord_dir = path_params['tfrecord_dir']
    log_dir = path_params['logs_dir']
    initial_weight = path_params['initial_weight']
    restore = solver_params['restore']
    classes = read_class_names(path_params['class_file'])
    class_num = len(classes)

    # 检查文件夹是否存在
    check_directory()

    # 配置GPU资源
    gpu_options = tf.ConfigProto(allow_soft_placement=True)
    gpu_options.gpu_options.allow_growth = True
    gpu_options.gpu_options.allocator_type = 'BFC'

    # 解析得到训练样本以及标注
    data = tfrecord.TFRecord()
    train_tfrecord = os.path.join(tfrecord_dir, "train.tfrecord")
    data_num = total_sample(train_tfrecord)
    batch_num = int(math.ceil(float(data_num) / batch_size))
    dataset = data.create_dataset(train_tfrecord, batch_num, batch_size=batch_size, is_shuffle=True)

    # 创建训练和验证数据迭代器
    iterator = dataset.make_one_shot_iterator()
    inputs, y_true_1, y_true_2, y_true_3 = iterator.get_next()

    inputs.set_shape([None, input_height, input_width, 3])
    y_true_1.set_shape([None, 13, 13, 3, 5 + class_num])
    y_true_2.set_shape([None, 26, 26, 3, 5 + class_num])
    y_true_3.set_shape([None, 52, 52, 3, 5 + class_num])
    y_true = [y_true_1, y_true_2, y_true_3]

    # 构建网络计算损失
    with tf.variable_scope('ODET'):
        model = Network(is_train=True)
        logits = model.forward(inputs)

    # 计算损失
    total_loss, iou_loss, conf_loss, class_loss = model.calc_loss(logits, y_true)
    #l2_loss = tf.losses.get_regularization_loss()
    #total_loss = loss_op[0] + loss_op[1] + loss_op[2]# + l2_loss

    # define training op
    global_step = tf.Variable(float(0), dtype=tf.float64, trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])
    learning_rate = tf.train.exponential_decay(solver_params['learning_rate'], global_step, solver_params['decay_steps'], solver_params['decay_rate'], staircase=True)
    #learning_rate = config_cosine_lr(batch_num, batch_size, num_epochs, 'learning_lr')
    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.937)
    optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.99)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(total_loss, global_step=global_step)
        # gvs = optimizer.compute_gradients(total_loss)
        # clip_grad_var = [gv if gv[0] is None else [tf.clip_by_norm(gv[0], 5.), gv[1]] for gv in gvs]
        # train_op = optimizer.apply_gradients(clip_grad_var, global_step=global_step)

    # 模型保存
    loader = tf.train.Saver()#tf.moving_average_variables())
    save_variable = tf.global_variables()
    saver = tf.train.Saver(save_variable, max_to_keep=1000)

    # 配置tensorboard
    tf.summary.scalar('learn_rate', learning_rate)
    tf.summary.scalar("iou_loss", iou_loss)
    tf.summary.scalar("conf_loss", conf_loss)
    tf.summary.scalar("class_loss", class_loss)
    tf.summary.scalar('total_loss', total_loss)

    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), flush_secs=60)

    # 开始训练
    with tf.Session(config=gpu_options) as sess:
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        if restore == True:
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            if ckpt and ckpt.model_checkpoint_path:
                stem = os.path.basename(ckpt.model_checkpoint_path)
                restore_step = int(stem.split('.')[0].split('-')[-1])
                start_step = restore_step
                sess.run(global_step.assign(restore_step))
                loader.restore(sess, ckpt.model_checkpoint_path)
                print('Restoreing from {}'.format(ckpt.model_checkpoint_path))
            else:
                print("Failed to find a checkpoint")

        summary_writer.add_graph(sess.graph)
        try:
            print('=> Restoring weights from: %s ... ' % initial_weight)
            loader.restore(sess, initial_weight)
        except:
            print('=> %s does not exist !!!' % initial_weight)
            print('=> Now it starts to train from scratch ...')

        print('\n----------- start to train -----------\n')
        for epoch in range(start_step + 1, num_epochs):
            train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss = [], [], [], []
            for index in tqdm(range(batch_num)):
                _, summary_, loss_, iou_loss_, confs_loss_, class_loss_, global_step_, lr = sess.run(
                    [train_op, summary_op, total_loss, iou_loss, conf_loss, class_loss, global_step, learning_rate])

                train_epoch_loss.append(loss_)
                train_epoch_iou_loss.append(iou_loss_)
                train_epoch_confs_loss.append(confs_loss_)
                train_epoch_class_loss.append(class_loss_)

                summary_writer.add_summary(summary_, global_step_)

            train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss = np.mean(train_epoch_loss), np.mean(train_epoch_iou_loss), np.mean(train_epoch_confs_loss), np.mean(train_epoch_class_loss)
            print("Epoch: {}, global_step: {}, lr: {:.8f}, total_loss: {:.3f}, iou_loss: {:.3f}, confs_loss: {:.3f}, class_loss: {:.3f}".format(epoch, global_step_, lr, train_epoch_loss, train_epoch_iou_loss, train_epoch_confs_loss, train_epoch_class_loss))
            snapshot_model_name = 'odet_train_mloss={:.4f}.ckpt'.format(train_epoch_loss)
            saver.save(sess, os.path.join(checkpoint_dir, snapshot_model_name), global_step=epoch)

        sess.close()

if __name__ == '__main__':
    train()