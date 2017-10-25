import tensorflow as tf

import loader, model
from loader import GEN_CONFIG

class _LoggerHook(tf.train.SessionRunHook):
    """Logs loss and runtime of a train op."""

    def begin(self):
        self._step = -1
        self._start_time = time.time()
        
    def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

    def after_run(self, run_context, run_values):
        if self._step % GEN_CONFIG['LOG_FREQUENCY'] == 0:
            current_time = time.time()
            duration = current_time - self._start_time
            self._start_time = current_time
            
            loss_value = run_values.results
            examples_per_sec = GEN_CONFIG['LOG_FREQUENCY'] * GEN_CONFIG['BATCH_SIZE'] / duration
            sec_per_batch = float(duration / GEN_CONFIG['LOG_FREQUENCY'])

            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
            print (format_str % (datetime.now(), self._step, loss_value, examples_per_sec, sec_per_batch))


def train_stock_cifar10():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        images, labels = loader.load_batch_cifar10()

        logits = model.stock_cifar10(images)
        loss = model.stock_cifar10_loss(logits, labels)
        train_op = model.stock_cifar10_train(loss, global_step)

        with tf.train.MonitoredTrainingSession(checkpoint_dir=CIFAR10_CONFIG['CHECKPOINT_DIR'],
                                               hooks=[tf.train.StopAtStepHook(last_step=CIFAR10_CONFIG['MAX_STEPS']),
                                                      tf.train.NanTensorHook(loss),
                                                      _LoggerHook()]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)
