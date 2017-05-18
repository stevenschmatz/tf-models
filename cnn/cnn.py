import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

def conv_2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding="SAME")
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def max_pool(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding="SAME")

def conv_net(x, weights, biases, dropout):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    
    with tf.name_scope('conv_1') as scope:
        conv_1 = conv_2d(x, weights['wc1'], biases['bc1'])
        conv_1 = max_pool(conv_1)

    with tf.name_scope('conv_2') as scope:
        conv_2 = conv_2d(conv_1, weights['wc2'], biases['bc2'])
        conv_2 = max_pool(conv_2)

    with tf.name_scope('fc_1') as scope:
        fc_1 = tf.reshape(conv_2, shape=[-1, weights['wd1'].get_shape().as_list()[0]])
        fc_1 = tf.nn.relu(tf.matmul(fc_1, weights['wd1']) + biases['bd1'])
        fc_1 = tf.nn.dropout(fc_1, dropout)
    
    with tf.name_scope('fc_2') as scope:
        return tf.matmul(fc_1, weights['wd2']) + biases['bd2']


weights = {
    # 5x5 conv, 1 input channel, 32 output channels
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='wc1'),

    # 5x5 conv, 32 input channels, 64 output channels
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc2'),

    # Fully connected 1, 7x7 with 64 channels in, 1024 out
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name='wd1'),

    # Fully connected 2, 1024 in, 10 out
    'wd2': tf.Variable(tf.random_normal([1024, 10]), name='wd2')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'bd2': tf.Variable(tf.random_normal([10]), name='bd2')
}

y_pred = conv_net(X, weights, biases, keep_prob)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.summary.scalar('cost', cost)
tf.summary.scalar('accuracy', accuracy)
merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("log_dir", tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    for i in range(500):
        batch_x, batch_y = mnist.train.next_batch(128)

        sess.run(optimizer, feed_dict={X: batch_x, y: batch_y, keep_prob: 0.75})

        if i % 10 == 0:
            cost_i, acc_i = sess.run([cost, accuracy], feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
            summary = sess.run(merged, feed_dict={X: batch_x, y: batch_y, keep_prob: 1.0})
            train_writer.add_summary(summary, i)

            print("Epoch {}, cost {}, acc {}".format(i, cost_i, acc_i))

    acc = sess.run(accuracy, feed_dict={X: mnist.test.images[:256], y: mnist.test.labels[:256], keep_prob: 1.0})
    print("Test accuracy: {}".format(acc))




