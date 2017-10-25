import tensorflow as tf

import loader, model
from loader import CIFAR10_CONFIG

def eval_stock_cifar10():
    with tf.Graph().as_default() as g:
        images, labels = loader.load_batch_cifar10_eval()
        logits = model.stock_cifar(images)
        top_k_op = tf.nn.in_top_k(logits, labels, 1)

        variable_averages = tf.train.ExponentialMovingAverage(float(CIFAR10_CONFIG['MOVING_AVERAGE_DECAY']))
        # defaults to tf.moving_average_variables() + tf.trainable_variables()
        variables_to_restore = variable_averages.variables_to_restore() 
        saver = tf.train.Saver(variables_to_restore)

        while True:
            #do some stuff (saver, top_k_op)
            if FLAGS.run_once:
                break
            time.sleep(5000)
