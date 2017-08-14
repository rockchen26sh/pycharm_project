from datetime import datetime
import math
import time
import tensorflow as tf

batch_size = 32
num_batch = 100

# 定义一个显示每一层结构的函数，展示每个卷积层或池化层输出tensor的尺寸。
# 函数接受一个tensor作为输入，并显示其名称（t.op.name）和tensor尺寸(t.get_shape.as_list())
def print_activations(t):
    print(t.op.name, ' ', t.get_shape().as_list())

# 定义一个inference函数,接受Images作为输入
# 返回最后一层pool5及parameters
# conv1卷积层1，

def inference(images):
    parameters = []

    with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 64],
                                                 dtype=tf.float32, stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(images, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)
        print_activations(conv1)
        parameters += [kernel, biases]

    with tf.name_scope('lrn1') as scope:
        lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name='lrn1')
        pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='VALID', name='pool1')
        print_activations(pool1)

    with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 192],
                                                 dtype=tf.float32, stddev=1e-1),name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, bias)
        conv2 = tf.nn.relu(bias, name=scope)
        parameters += [kernel, biases]
        print_activations(conv2)

    with tf.name_scope('lrn2') as scope:
        lrn2 = tf.nn.lrn(conv2, )