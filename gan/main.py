import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

def count_params():
    from functools import reduce
    size = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
    return sum(size(v) for v in tf.trainable_variables())

X = tf.placeholder(tf.float32, [None, 784])

d_w1 = tf.Variable(tf.random_normal([784, 256]))
d_b1 = tf.Variable(tf.constant(0.1, shape=[256]))
d_w2 = tf.Variable(tf.random_normal([256, 128]))
d_b2 = tf.Variable(tf.constant(0.1, shape=[128]))
d_w3 = tf.Variable(tf.random_normal([128, 1]))
d_b3 = tf.Variable(tf.constant(0.1, shape=[1]))

d_vars = [d_w1, d_b1, d_w2, d_b2, d_w3, d_b3]

def discriminator(x):
    layer_1 = tf.nn.relu(tf.matmul(x, d_w1) + d_b1)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, d_w2) + d_b2)
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, d_w3) + d_b3)

    return layer_3 

z = tf.placeholder(tf.float32, shape=[None, 100])

g_w1 = tf.Variable(tf.random_normal([100, 128]))
g_b1 = tf.Variable(tf.constant(0.1, shape=[128]))
g_w2 = tf.Variable(tf.random_normal([128, 256]))
g_b2 = tf.Variable(tf.constant(0.1, shape=[256]))
g_w3 = tf.Variable(tf.random_normal([256, 784]))
g_b3 = tf.Variable(tf.constant(0.1, shape=[784]))

g_vars = [g_w1, g_b1, g_w2, g_b2, g_w3, g_b3]

def generator(z):
    layer_1 = tf.nn.relu(tf.matmul(z, g_w1) + g_b1)
    layer_2 = tf.nn.relu(tf.matmul(layer_1, g_w2) + g_b2)
    layer_3 = tf.nn.tanh(tf.matmul(layer_2, g_w3) + g_b3)

    return layer_3 

def sample_z(m, n):
    return np.random.uniform(-1., 1., size=[m,n])


g_sample = generator(z)
d_fake = discriminator(g_sample)
d_real = discriminator(X)

d_loss = -tf.reduce_mean(tf.log(d_real) + tf.log(1 - d_fake))
g_loss = -tf.reduce_mean(tf.log(d_fake))

d_train = tf.train.AdamOptimizer().minimize(d_loss, var_list=d_vars)
g_train = tf.train.AdamOptimizer().minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10000):
        batch_xs, _ = mnist.train.next_batch(256)

        _, d_loss_curr = sess.run([d_train, d_loss], feed_dict={X: batch_xs, z: sample_z(256, 100)})
        _, g_loss_curr = sess.run([g_train, g_loss], feed_dict={z: sample_z(256, 100)})

        if epoch % 10 == 0:
            print("Epoch {}, d loss {}, g loss {}".format(epoch, d_loss_curr, g_loss_curr))

