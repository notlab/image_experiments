import os

import numpy as np
import tensorflow as tf
from PIL import Image

from models.stylenet import vgg16

x_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf.float32)
img = tf.get_variable('image', [1, 500, 500, 3], initializer=x_init)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    actual_image = sess.run(img)

    vgg = vgg16.Vgg16('fc3', True, sess)


    #img = Image.fromarray(actual_image[0], 'RGB')
    #img.show()
