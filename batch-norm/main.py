import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

def dense(x, size, scope):
    return tf.contrib.layers.fully_connected(x, size, activation_fn=None, scope=scope)

def dense_batch_relu(x, phase, scope):
    with tf.variable_scope(scope):
        h1 = tf.contrib.layers.fully_connected(x, 100, activation_fn=None, scope='dense')
        h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn')
        return tf.nn.relu(h2, name='relu')

tf.reset_default_graph()

x = tf.placeholder(tf.float32, [None, 784], name='x')
y = tf.placeholder(tf.float32, [None, 10], name='y')
phase = tf.placeholder(tf.bool, name='phase')

h1 = dense_batch_relu(x, phase, 'layer_1')
h2 = dense_batch_relu(h1, phase, 'layer_2')
logits = dense(h2, 10, 'logits')

with tf.name_scope('accuracy'):
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1)), tf.float32))

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))

def feed_dict_status(train=True):
    image_set = mnist.train if train else mnist.test
    return {x: image_set.images, y: image_set.labels, phase: train}
    

def train():
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    iterep = 500
    for i in range(iterep * 30):
        x_train, y_train = mnist.train.next_batch(100)
        sess.run(optimize, feed_dict={x: x_train, y: y_train, phase:True})
        
        if i % iterep == 0:
            acc_train, loss_train = sess.run([accuracy, loss], feed_dict=feed_dict_status(True))
            print("Training: acc {}, loss {}".format(acc_train, loss_train))
            acc_test, loss_test = sess.run([accuracy, loss], feed_dict=feed_dict_status(False))
            print("Testing: acc {}, loss {}".format(acc_test, loss_test))

train()
