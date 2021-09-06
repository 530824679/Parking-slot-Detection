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
import yaml
import shutil
import logging
import argparse
import numpy as np
import tensorflow as tf
import glog as log
from pathlib import Path

from utils.general import get_latest_run, check_file, increment_dir, colorstr
from model.network import Network

from cfg.config import path_params, model_params, solver_params
from data import dataset, tfrecord
from utils.data_utils import *

logger = logging.getLogger(__name__)

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

def train(hyp, opt):
    logger.info(colorstr('hyperparameters: ') + ', '.join(f'{k}={v}' for k, v in hyp.items()))
    log_dir = Path(opt.log_dir)

    # Directories
    wdir = log_dir / 'weights'
    os.makedirs(wdir, exist_ok=True)

    last = os.path.join(opt.model, 'last.ckpt')
    best = os.path.join(opt.model, 'best.ckpt')
    results_file = str(log_dir / 'results.txt')

    epochs, batch_size, weights = opt.epochs, opt.batch_size, opt.weights

    # Save run settings
    with open(log_dir / 'hyp.yaml', 'w') as f:
        yaml.dump(hyp, f, sort_keys=False)
    with open(log_dir / 'opt.yaml', 'w') as f:
        yaml.dump(vars(opt), f, sort_keys=False)

    # Configure GPU
    gpu_options = tf.ConfigProto(allow_soft_placement=True)
    gpu_options.gpu_options.allow_growth = True
    gpu_options.gpu_options.allocator_type = 'BFC'

    # Data dict
    with open(opt.data) as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)

    train_path = data_dict['train']
    test_path = data_dict['val']

    # Add hyperparameters
    loggers = None
    opt.hyp = hyp

    # number of classes
    num_classes = int(data_dict['num_classes'])

    # class names
    names = data_dict['names']
    assert len(names) == num_classes, '%g names found for nc=%g dataset in %s' % (len(names), nc, opt.data)  # check

    # Create model
    model = Network(opt.config, ch=3, num_classes=num_classes, anchors=hyp.get('anchors'))






    # Freeze
    freeze = []  # parameter names to freeze (full or partial)
    for k, v in model.named_parameters():
        v.requires_grad = True  # train all layers
        if any(x in k for x in freeze):
            print('freezing %s' % k)
            v.requires_grad = False

    # Optimizer
    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    hyp['weight_decay'] *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f"Scaled weight_decay = {hyp['weight_decay']}")

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay

    optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)

    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    logger.info('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    lf = one_cycle(1, hyp['lrf'], epochs)  # cosine 1->hyp['lrf']
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)








    # 检查文件夹是否存在
    check_directory()

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./weights/odet.ckpt', help='initial weights path')
    parser.add_argument('--config', type=str, default='./config/odet.yaml', help='model yaml path')
    parser.add_argument('--data', type=str, default='./config/voc.yaml', help='data yaml path')
    parser.add_argument('--hyp', type=str, default='./config/hyp.yaml', help='hyperparameters yaml path')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=8, help='total batch size for all GPUs')
    parser.add_argument('--image-size', nargs='+', type=int, default=[640, 640], help='[train, test] image sizes')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--noautoanchor', action='store_true', help='disable autoanchor check')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--workers', type=int, default=0, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--upload_dataset', action='store_true', help='Upload dataset as W&B artifact table')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval for W&B')
    parser.add_argument('--save_period', type=int, default=-1, help='Log model after every "save_period" epoch')
    parser.add_argument('--logdir', type=str, default='/home/chenwei/HDD/Project/ml/output/tensorboard', help='logging directory')
    parser.add_argument('--model', default='/home/chenwei/HDD/Project/ml/model', help='where the model is going to save')
    parser.add_argument('--checkpoints', default='/home/chenwei/HDD/Project/ml/checkpoints', help='where the model checkpoints is going to save')
    opt = parser.parse_args()

    if opt.resume:
        ckpt = opt.resume if isinstance(opt.resume, str) else get_latest_run()
        log_dir = Path(ckpt).parent.parent
        assert os.path.isfile(ckpt), 'ERROR: --resume checkpoint does not exist'
        with open(log_dir / 'opt.yaml') as f:
            opt = argparse.Namespace(**yaml.load(f, Loader=yaml.SafeLoader))
        logger.info('Resuming training from %s' % ckpt)
    else:
        opt.data, opt.config, opt.hyp = check_file(opt.data), check_file(opt.config), check_file(opt.hyp)  # check files
        assert len(opt.config) or len(opt.weights), 'either --cfg or --weights must be specified'
        opt.image_size.extend([opt.image_size[-1]] * (2 - len(opt.image_size)))
        log_dir = increment_dir(Path(opt.logdir) / 'exp', opt.name)

    # Hyperparameters
    with open(opt.hyp) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

    # Logger
    logger.info(opt)
    prefix = colorstr('tensorboard: ')
    logger.info('Start Tensorboard with "tensorboard --logdir %s", view at http://localhost:6006/' % opt.logdir)

    # Tensorboard
    #tb_writer = None
    #tb_writer = SummaryWriter(log_dir=log_dir)

    # Train
    train(hyp, opt)