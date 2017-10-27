import time
from datetime import datetime
import os

import tensorflow as tf

from models.common import ROOT_DIR, CONFIG
import models.cifar10.loader as loader
import models.cifar10.model as model

GEN_CONFIG = CONFIG['GENERAL']
CIFAR10_CONFIG = CONFIG['CIFAR_10']

class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime of a train op."""

    def __init__(self, loss):
        self._loss = loss
    
    def begin(self):
        self._step = -1
        self._start_time = time.time()
        
    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(self._loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        log_freq = int(GEN_CONFIG['LOG_FREQUENCY'])
        batch_size = int(GEN_CONFIG['BATCH_SIZE'])
        if self._step % log_freq == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            
            loss_value = run_values.results
            examples_per_sec = log_freq * batch_size / duration
            sec_per_batch = float(duration / log_freq)

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))


def train_stock_cifar10():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        images, labels = loader.load_train_batch()

        logits = model.stock_cifar10(images)
        loss = model.stock_cifar10_loss(logits, labels)
        train_op = model.stock_cifar10_train(loss, global_step)

        checkpoint_dir = os.path.join(ROOT_DIR, CIFAR10_CONFIG['CHECKPOINT_DIR'])

        with tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir,
                                               hooks=[tf.train.StopAtStepHook(last_step=int(CIFAR10_CONFIG['MAX_STEPS'])),
                                                      tf.train.NanTensorHook(loss),
                                                      _LoggerHook(loss)]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
