import tensorflow as tf

from loader import GEN_CONFIG, CIFAR10_CONFIG

def _get_tn_var(name, shape, stddev, reg=None):
    '''
    Get a variable with truncated normal initializer and optional l2 regularization. 
    '''
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)

    if reg is not None:
        weight_penalty = tf.multiply(tf.nn.l2_loss(var), reg, name='weight_loss')
        tf.add_to_collection('losses', weight_penalty)
        
    return var

def _get_kernel(name, shape, stddev, reg=None):
    '''
    Alias for _get_tn_var. 
    '''
    return _get_tn_var(name, shape, stddev, reg=reg)

def stock_cifar10(inputs):
    '''
    Args:
      inputs: expects input with shape [ batch_size, IMG_SIZE, IMG_SIZE, 3 ]. 
              Note inputs must be sqaure. 
    '''
    # Conv 1
    # output shape = [N, IMG_SIZE, IMG_SIZE, 64]
    with tf.variable_scope('conv1') as scope:
        kernel = _get_kernel('weights', shape=[5, 5, 3, 64], stddev=5e-2, reg=0.0)
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.0))
        pre_act = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_act, name=scope.name)

    # Pool 1
    # if C = ceil(IMG_SIZE / 2), then output shape = [N, C, C, 64] 
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    # Norm 1
    # no shape change
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # Conv 2
    # output shape = [N, C, C, 64]
    with tf.variable_scope('conv2') as scope:
        kernel = _get_kernel('weights', shape=[5, 5, 64, 64], stddev=5e-2, reg=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1))
        pre_act = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_act, name=scope.name)

    # Norm 2
    # no shape change
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    
    # Pool 2
    # if C0 = ceil(ceil(IMG_SIZE / 2) / 2) then output shape = [N, C0, C0, 64]
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # Fully connected 1
    # reshapes input to [N, C0 * C0 * 64], outputs shape [384]
    with tf.variable_scope('local3') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool2, [int(GEN_CONFIG['BATCH_SIZE']), -1])
        dim = reshape.get_shape()[1].value
        weights = _get_tn_var('weights', shape=[dim, 384], stddev=0.04, reg=0.004)
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # Fully connected 2
    # input [348] -> output [192]
    with tf.variable_scope('local4') as scope:
        weights = _get_tn_var('weights', shape=[384, 192], stddev=0.04, reg=0.004)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # Compute unscaled logits. Return unscaled logits (cross-entropy loss function will softmax our
    # logits for efficiecy). 
    with tf.variable_scope('softmax_linear') as scope:
        weights = _get_tn_var('weights', [192, int(CIFAR10_CONFIG['NUM_CLASSES'])], stddev=1/192.0, reg=0.0)
        biases = tf.get_variable('biases', [int(CIFAR10_CONFIG['NUM_CLASSES'])], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

    return softmax_linear

def stock_cifar10_loss(logits, labels):
    # Calculate the average cross entropy loss across the batch.
    labels = tf.cast(labels, tf.int64)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    ce_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    # The total loss is the error function (cross-entropy) plus the regularization terms from the
    # conv layer filters. Those were stored in the 'losses' collection. (See _get_kernel())
    tf.add_to_collection('losses', ce_mean)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def stock_cifar10_train(total_loss, global_step):
    batches_per_epoch = int(GEN_CONFIG['EPOCH_SIZE_TRAIN']) / int(GEN_CONFIG['BATCH_SIZE'])
    
    # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
    # so if you want 5 decays, and you're going to take 5000 steps, decay_steps should be 5000/5 = 1000
    decay_steps = int(batches_per_epoch * int(GEN_CONFIG['EPOCHS_PER_DECAY_STEP']))

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(float(GEN_CONFIG['INITIAL_LEARNING_RATE']),
                                    global_step,
                                    decay_steps,
                                    float(GEN_CONFIG['LEARNING_RATE_DECAY_FACTOR']),
                                    staircase=True)

    opt = tf.train.GradientDescentOptimizer(lr)
    grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    variable_averages = tf.train.ExponentialMovingAverage(float(CIFAR10_CONFIG['MOVING_AVERAGE_DECAY']), global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Tie off the training operations.
    with tf.control_dependencies([apply_gradient_op, variables_averages_op]): 
        end_op = tf.no_op(name='train')
    
    return end_op

    
