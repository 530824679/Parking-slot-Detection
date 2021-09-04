import os
import time
import math
import platform
import logging
import tensorflow as tf

from copy import deepcopy
from contextlib import contextmanager
from tensorflow.python.client import device_lib

logger = logging.getLogger(__name__)


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    s = f'odet 🚀 tensorflow {tf.__version__} '  # string
    cpu = device.lower() == 'cpu'
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    elif device:  # non-cpu device requested
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert tf.test.is_built_with_cuda(), f'CUDA unavailable, invalid device {device} requested'  # check availability

    cuda = not cpu and tf.test.is_built_with_cuda()
    if cuda:
        local_device_protos = device_lib.list_local_devices()
        num_gpus = sum([1 for d in local_device_protos if d.device_type == 'GPU'])
        if num_gpus > 1 and batch_size:
            assert batch_size % num_gpus == 0, f'batch-size {batch_size} not multiple of GPU count {num_gpus}'
        space = ' ' * len(s)
        for i, d in enumerate(device.split(',') if device else range(num_gpus)):
            s += f"{'' if i == 0 else space}CUDA:{d} ({local_device_protos[i].name}, {local_device_protos[i].memory_limit / 1024 ** 2}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    logger.info(s.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else s)  # emoji-safe
    return tf.device('cuda:0' if cuda else 'cpu')