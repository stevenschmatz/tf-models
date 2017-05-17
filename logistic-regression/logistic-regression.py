"""
Logistic Regression on the Iris dataset.

Author: @stevenschmatz
Repo: https://github.com/stevenschmatz/tf-models/
"""

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Define hyperparameters
learning_rate = 0.01
training_epochs = 10000

# Define inputs and outputs
with tf.name_scope("input-output") as scope:
    X = tf.placeholder(tf.float32, [None, 4], name='input')
    y = tf.placeholder(tf.float32, [None, 3], name='output')

# Define model
with tf.name_scope("model") as scope:
    W = tf.Variable(tf.truncated_normal([4, 3]), name='weight')
    b = tf.Variable(tf.truncated_normal([3]), name='bias')
    y_hat = tf.nn.softmax(tf.matmul(X, W) + b, name='prediction')

# Define training
with tf.name_scope("train") as scope:
    loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_hat), reduction_indices=1))
    train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Bookkeeping for TensorBoard
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', accuracy)
summary_op = tf.summary.merge_all()

# Define data
iris = load_iris()
X_data = iris.data
y_data = np.array([[int(val == index) for index in range(3)] for val in iris.target]) # one hot
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.33)

# Run training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    writer = tf.summary.FileWriter('log/logistic-regression', sess.graph)

    for epoch in range(training_epochs):
        _, cost, summary_train = sess.run([train, loss, summary_op], feed_dict={X: X_train, y: y_train})

        if epoch % (training_epochs // 100) == 0:
            writer.add_summary(summary_train, epoch)
            print("Epoch {}, cross entropy = {}".format(epoch, cost))

    print("Accuracy:", accuracy.eval({X: X_test, y: y_test}))

