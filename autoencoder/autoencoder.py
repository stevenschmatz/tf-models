import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

class Autoencoder(object):
    '''A basic autoencoder with six fully connected layers.'''

    def __init__(self):
        self.sess = tf.Session()

    def build_model(self):

        # Network parameters
        n_input = 784
        n_hidden_1 = 256
        n_hidden_2 = 128
        n_hidden_3 = 64

        weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h3': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        }

        biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b2': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b3': tf.Variable(tf.random_normal([n_input]))
        }

        def encode(x):
            layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['encoder_h1']) + biases['encoder_b1'])
            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['encoder_h2']) + biases['encoder_b2'])
            layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['encoder_h3']) + biases['encoder_b3'])

            return layer_3

        def decode(x):
            layer_1 = tf.nn.sigmoid(tf.matmul(x, weights['decoder_h1']) + biases['decoder_b1'])
            layer_2 = tf.nn.sigmoid(tf.matmul(layer_1, weights['decoder_h2']) + biases['decoder_b2'])
            layer_3 = tf.nn.sigmoid(tf.matmul(layer_2, weights['decoder_h3']) + biases['decoder_b3'])

            return layer_3

        self.X = tf.placeholder(tf.float32, [None, n_input])

        encode_op = encode(self.X)
        decode_op = decode(encode_op)

        y_pred = decode_op 
        y_true = self.X

        return y_pred, y_true


    def train(self, save=True, learning_rate=0.01, training_epochs=20, batch_size=256, display_step=1):
        '''Trains the network with the given hyperparameters, and saves to a checkpoint directory.'''
        
        self.y_pred, self.y_true = self.build_model()

        cost = tf.reduce_mean(tf.square(self.y_pred - self.y_true))
        train = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

        saver = tf.train.Saver()

        if os.path.exists('checkpoint') and save:  
            saver.restore(self.sess, tf.train.latest_checkpoint('checkpoint'))
        else:
            os.mkdir('checkpoint')
            self.sess.run(tf.global_variables_initializer())

            total_batch = int(mnist.train.num_examples/batch_size)

            for epoch in range(training_epochs):
                for i in range(total_batch):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    _, c = self.sess.run([train, cost], feed_dict={self.X: batch_xs})

                if epoch % display_step == 0:
                    print("Step {}, cost {}".format(epoch, c))

            if save:
                saver.save(self.sess, "checkpoint/model.ckpt")

    def test(self, examples_to_show=10):
        '''Shows the first ten images of MNIST, and their respective reconstructions.'''

        encode_decode = self.sess.run(self.y_pred, feed_dict={self.X: mnist.test.images[:examples_to_show]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for i in range(examples_to_show):
            a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))
        f.show()
        plt.draw()
        plt.waitforbuttonpress()


model = Autoencoder()
model.train()
model.test()
