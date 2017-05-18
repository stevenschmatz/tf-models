import tensorflow as tf
import numpy as np

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

learning_rate = 0.5

class Generator(object):
    def __init__(self):

    @property
    def prior():
        return tf.random_normal

    def generate_samples_op(batch_size=256):
        noise = self.prior([batch_size])
        

    def descend_gradient(discriminator):
        samples = self.generate_samples()
        cost = tf.reduce_mean(tf.log(1-discriminator(self.generate_samples())))
        train = tf.train.AdamOptimizer(learning_rate)

        return train


