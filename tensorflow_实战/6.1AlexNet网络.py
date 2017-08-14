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