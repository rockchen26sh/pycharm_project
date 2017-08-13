#enconding -* utf-8 *-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

#定义卷积层函数
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

#定义池化层函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') #strides设为横竖方向以2为步长

#input
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1]) #-1代表数量不固定

#定义卷积层
w_conv = weight_variable([5,5,1,32])
b_conv = bias_variable([32])
h_conv = tf.nn.relu(conv2d(x_image, w_conv) + b_conv)
h_pool = max_pool_2x2(h_conv)

#全连接层
w_fc1 = weight_variable([14*14*32, 1024])
b_fc1 = bias_variable([1024])
h_pool_flat = tf.reshape(h_pool, [-1, 14*14*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, w_fc1) + b_fc1)

#dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

#loss function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

#train_step
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

#accuracy
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1],keep_prob:1.0})
            print ('step %d, training_accuracy: %f'%(i, train_accuracy))

        sess.run(train_step, feed_dict={x:batch[0], y_:batch[1],keep_prob:0.5})

        if i%1000 == 0:
            test_accuracy = sess.run(accuracy, feed_dict= {x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0})
            print('step %d, test_accuracy: %f' % (i,  test_accuracy))
