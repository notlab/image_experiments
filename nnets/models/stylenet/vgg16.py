import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from imagenet_classes import class_names

#             layer name --> filter shape
VGG_ARCH = [ ('prep', None), # preprocess layer, subtract ImageNet image means
             ('conv1_1', [3, 3, 3, 64]),
             ('conv1_2', [3, 3, 64, 64]),
             ('pool1', None),
             ('conv2_1', [3, 3, 64, 128]),
             ('conv2_2', [3, 3, 128, 128]),
             ('pool2', None),
             ('conv3_1', [3, 3, 128, 256]),
             ('conv3_2', [3, 3, 256, 256]),
             ('conv3_3', [3, 3, 256, 256]),
             ('pool3', None),
             ('conv4_1', [3, 3, 256, 512]),
             ('conv4_2', [3, 3, 512, 512]),
             ('conv4_3', [3, 3, 512, 512]),
             ('pool4', None),
             ('conv5_1', [3, 3, 512, 512]),
             ('conv5_2', [3, 3, 512, 512]),
             ('conv5_3', [3, 3, 512, 512]),
             ('pool5', None),
             ('fc1', [None, 4096]), # first fully connected layer must reshape pooling output
             ('fc2', [4096, 4096]),
             ('fc3', [4096, 1000]) ]


class vgg16:

    def __init__(self, image_placeholder, weights=None, sess=None):
        self.net = image_placeholder
        self.parameters = []
        self._revive()
        if weights is not None and sess is not None:
            self._load_weights(weights, sess)

    def _revive(self):
        for layer_arch in VGG_ARCH:
            layer_name = layer_arch[0]
            if layer_name[:4] == 'prep':
                self._preprocess()
            elif layer_name[:4] == 'conv':
                self._conv_layer(layer_arch)
            elif layer_name[:4] == 'pool':
                self._pool_layer(layer_arch)
            elif layer_name[:3] == 'fc1':
                self._fc_transition(layer_arch)
            elif layer_name[:3] in set('fc2', 'fc3'):
                self._fc_layer(layer_arch)
            else:
                raise ValueError("Unknown layer type: %s" % layer_name)                
                            
    def _preprocess(self):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            self.net = self.net - mean

    def _conv_layer(self, layer_arch):
        name, filter_shape = layer_arch[0], layer_arch[1]
        bias_shape = filter_shape[-1]

        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(filter_shape, dtype=tf.float32, stddev=1e-1),
                                 trainable=False, name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=bias_shape, dtype=tf.float32),
                                 trainable=False, name='biases')
            self.net = tf.nn.conv2d(self.net, kernel, [1, 1, 1, 1], padding='SAME')
            self.net = tf.nn.bias_add(conv, biases)
            self.net = tf.nn.relu(self.net, name=scope)
            self.parameters += [kernel, biases]

    def _pool_layer(self, layer_arch):
        self.net = tf.nn.max_pool(self.net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_arch[0])

    def _fc_transition(self, layer_arch):
        weight_shape = layer_arch[1]
        # gets the dimension of the incoming connections, sets height of weight matrix
        weight_shape[0] = int(np.prod(self.net.get_shape()[1:]))
        bias_shape = weight_shape[1]
        
        # transition fc layer is also first fc layer, hence name is 'fc1'
        with tf.name_scope('fc1') as scope: 
            fc1_W = tf.Variable(tf.truncated_normal(weight_shape, dtype=tf.float32, stddev=1e-1),
                                trainable=False,
                                name='weights')
            fc1_b = tf.Variable(tf.constant(1.0, shape=bias_shape, dtype=tf.float32),
                                trainable=False,
                                name='biases')
            self.net = tf.reshape(self.net, [-1, shape[0]])
            self.net = tf.nn.bias_add(tf.matmul(self.net, fc1_W), fc1_b)
            self.net = tf.nn.relu(self.net)
            self.parameters += [fc1_W, fc1_b]

    def _fc_layer(self, layer_arch):
        name, weight_shape = layer_arch[0], layer_arch[1]
        bias_shape = weight_shape[1]

        with tf.name_scope(name) as scope:
            fc2_W = tf.Variable(tf.truncated_normal(weight_shape, dtype=tf.float32, stddev=1e-1),
                                trainable=False,
                                name='weights')
            fc2_b = tf.Variable(tf.constant(1.0, shape=bias_shape, dtype=tf.float32),
                                trainable=False,
                                name='biases')
            self.net = tf.nn.bias_add(tf.matmul(self.net, fc2w), fc2b)
            self.net = tf.nn.relu(self.net)
            self.parameters += [fc2_W, fc2_b]

    def _load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))
