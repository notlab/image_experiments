import tensorflow as tf

def _get_kernel(name, shape, stddev, reg=None):
    initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = tf.get_variable(name, shape, initializer)

    if reg is not None:
        weight_penalty = tf.multiply(tf.nn.l2_loss(var, reg, name='weight_loss'))
        tf.add_to_collection('losses', weight_penalty)
        
    return var
    

def conv_net_1(inputs):
    """
    Args:
      inputs: expects input with shape [ batch_size, IMG_SIZE, IMG_SIZE, 3 ]. 
              Note inputs must be sqaure. 
    """
    # Conv 1
    # output shape = [N, IMG_SIZE, IMG_SIZE, 64]
    with tf.variable_scope('conv1') as scope:
        kernel = _get_kernel('weights', shape=[5, 5, 3, 64], stddev=5e-2, reg=0.0)
        conv = tf.nn.conv2d(inputs, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], tf.constant_initializer(0.0))
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
        kernel = _get_kernel('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.get_variable('biases', [64], tf.constant_initializer(0.1))
        pre_act = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_act, name=scope.name)
