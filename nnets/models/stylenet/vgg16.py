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


class Vgg16:

    def __init__(self, depth, weights=None, sess=None):
        '''
        Parameters: 
          depth: which layer to stop at when constructing vgg net. E.g. if "conv4_3" is passed,
                 this will be the output layer of the constructed Vgg16. Later layers won't be added.
          weights: the weights file from which to load pre-trained weights.
          sess: A TensorFlow session object to use when loading weights. 
        '''
        self.depth = depth
        if weights is not None and sess is not None:
            self._load_weights(weights, sess)

    def run_once(self, image):
        for layer_arch in VGG_ARCH:
            layer_name = layer_arch[0]
            # stop building network if we've reached desired depth.
            if layer_name == self.depth:
                break
            
            if layer_name[:4] == 'prep':
                net = self._preprocess(image)
            elif layer_name[:4] == 'conv':
                net = self._conv_layer(layer_arch, net)
            elif layer_name[:4] == 'pool':
                net = self._pool_layer(layer_arch, net)
            elif layer_name[:3] == 'fc1':
                net = self._fc_transition(layer_arch, net)
            elif layer_name[:3] in set('fc2', 'fc3'):
                net = self._fc_layer(layer_arch, net)
            else:
                raise ValueError("Unknown layer type: %s" % layer_name)

            return net
                            
    def _preprocess(self, image):
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            return image - mean

    def _conv_layer(self, layer_arch, net):
        name, filter_shape = layer_arch[0], layer_arch[1]
        bias_shape = filter_shape[-1]

        with tf.name_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(filter_shape, dtype=tf.float32, stddev=1e-1),
                                 trainable=False, name='weights')
            biases = tf.Variable(tf.constant(0.0, shape=bias_shape, dtype=tf.float32),
                                 trainable=False, name='biases')
            net = tf.nn.conv2d(net, kernel, [1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(conv, biases)
            net = tf.nn.relu(net, name=scope)

        return net

    def _pool_layer(self, layer_arch, net):
        return tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_arch[0])

    def _fc_transition(self, layer_arch, net):
        weight_shape = layer_arch[1]
        # gets the dimension of the incoming connections, sets height of weight matrix
        weight_shape[0] = int(np.prod(net.get_shape()[1:]))
        bias_shape = weight_shape[1]
        
        # transition fc layer is also first fc layer, hence name is 'fc1'
        with tf.name_scope('fc1') as scope: 
            fc1_W = tf.Variable(tf.truncated_normal(weight_shape, dtype=tf.float32, stddev=1e-1),
                                trainable=False,
                                name='weights')
            fc1_b = tf.Variable(tf.constant(1.0, shape=bias_shape, dtype=tf.float32),
                                trainable=False,
                                name='biases')
            net = tf.reshape(net, [-1, shape[0]])
            net = tf.nn.bias_add(tf.matmul(net, fc1_W), fc1_b)
            net = tf.nn.relu(net)

        return net

    def _fc_layer(self, layer_arch, net):
        name, weight_shape = layer_arch[0], layer_arch[1]
        bias_shape = weight_shape[1]

        with tf.name_scope(name) as scope:
            fc2_W = tf.Variable(tf.truncated_normal(weight_shape, dtype=tf.float32, stddev=1e-1),
                                trainable=False,
                                name='weights')
            fc2_b = tf.Variable(tf.constant(1.0, shape=bias_shape, dtype=tf.float32),
                                trainable=False,
                                name='biases')
            net = tf.nn.bias_add(tf.matmul(net, fc2w), fc2b)
            net = tf.nn.relu(net)

    def _load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))
