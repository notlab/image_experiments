import tensorflow as tf

import models.stylenet.vgg16

def Stylizer:

    def __init__(self, target_image, style_image, vgg):
        assert style_image.shape == target_image.shape
        self.target_image = target_image
        self.style_image = style_image
        self.vgg = vgg

    def train(self, output, global_step):
        loss = self.total_loss(output)
        lr = tf.train.exponential_decay(0.1, global_step, decay_steps, 0.1, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        return apply_gradient_op
        
    def total_loss(self, output):
        activations = self.vgg.run_once(output)
        style, target = self.vgg.run_once(self.target_image), self.vgg.run_once(self.style_image)
        total_loss = tf.add(content_loss(target, activations), style_loss(style, activations))

    def content_loss(self, target, style):
        dim = np.prod(style.shape[1:])
        target = tf.reshape(target, [-1, dim])
        style = tf.reshape(style, [-1, dim])
        return tf.losses.mean_squared_error(target, style)

    def style_loss(self, target, style):
        tar_gram = self.gram_matrix(target)
        sty_gram = self.gram_matrix(style)
        return tf.losses.mean_squared_error(tar_gram, sty_gram)

    def gram_matrix(self, variable):
        batch_size, height, width, channels = tf.unstack(tf.shape(variable))
        denominator = tf.to_float(height * width)
        variable = tf.reshape(variabel, tf.stack([batch_size, height * width, channels]))
        matrix = tf.matmul(variable, variable, adjoint_a=True)
        return matrix / denominator
        

