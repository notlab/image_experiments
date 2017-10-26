import tensorflow as tf

import loader, model
from loader import GEN_CONFIG, CIFAR10_CONFIG

def _do_eval_stock_cifar10(saver, top_k_op):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CIFAR10_CONFIG['CHECKPOINT_DIR'])
        
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No trained models present (no checkpoints found).')
            return

        # Fire up queue runners
        coord = tf.train.Coordinator()
        threads = []
        
        try:
            for queue_runner in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_examples = int(GEN_CONFIG['EPOCH_SIZE_EVAL'])
                batch_size = int(GEN_CONFIG['BATCH_SIZE'])
                num_iterations = int(math.ceil(num_examples / batch_size))
                true_count = 0
                step = 0

                while step < num_iterations and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    true_count += np.sum(predictions)
                    step += 1

                precision = true_count / num_examples
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

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
