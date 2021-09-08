
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class Optimizer(object):
    def __init__(self, optimizer_method='adam'):
        self.optimizer_method = optimizer_method

    def __call__(self):
        if self.optimizer_method == 'adam':
            return tf.keras.optimizers.Adam()
        elif self.optimizer_method == 'rmsprop':
            return tf.keras.optimizers.RMSprop()
        elif self.optimizer_method == 'sgd':
            return tf.keras.optimizers.SGD()
        else:
            raise ValueError('Unsupported optimizer {}'.format(self.optimizer_method))

class Cosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, total_steps, params):
        # create the cosine learning rate with linear warmup
        super(Cosine, self).__init__()
        self.total_steps = total_steps
        self.params = params

    def __call__(self, global_step):
        init_lr = self.params['init_learning_rate']
        warmup_lr = self.params['warmup_learning_rate'] if 'warmup_learning_rate' in self.params else 0.0
        warmup_steps = self.params['warmup_steps']
        assert warmup_steps < self.total_steps, "warmup {}, total {}".format(warmup_steps, self.total_steps)

        linear_warmup = warmup_lr + tf.cast(global_step, tf.float32) / warmup_steps * (init_lr - warmup_lr)
        cosine_learning_rate = init_lr * (
                    tf.cos(np.pi * (global_step - warmup_steps) / (self.total_steps - warmup_steps)) + 1.0) / 2.0
        learning_rate = tf.where(global_step < warmup_steps, linear_warmup, cosine_learning_rate)
        return learning_rate