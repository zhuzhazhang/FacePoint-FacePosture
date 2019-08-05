import tensorflow as tf
# import numpy as np
from read import read_data
from read_test import read_test
import matplotlib.pyplot as plt


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# 1.训练的数据
# Make up some real data
x_train, y_train, x_test, y_test = read_data("data_train_4.csv")
x_test_cnn = read_test("predict.csv")

# 2.定义节点准备接收数据
# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 10])
ys = tf.placeholder(tf.float32, [None, 3])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs
l1 = add_layer(xs, 10, 60, activation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 60, 60, activation_function=tf.nn.sigmoid)
# l3 = add_layer(l2, 26, 24, activation_function=tf.nn.sigmoid)
# l4 = add_layer(l3, 24, 22, activation_function=tf.nn.sigmoid)
# l5 = add_layer(l4, 22, 20, activation_function=tf.nn.sigmoid)
# l6 = add_layer(l5, 20, 18, activation_function=tf.nn.sigmoid)
# l7 = add_layer(l6, 18, 16, activation_function=tf.nn.sigmoid)
# l8 = add_layer(l7, 16, 14, activation_function=tf.nn.sigmoid)
# l9 = add_layer(l8, 14, 12, activation_function=tf.nn.sigmoid)
# l10 = add_layer(l9, 12, 10, activation_function=tf.nn.sigmoid)
# l11 = add_layer(l10, 10, 8, activation_function=tf.nn.sigmoid)
# l12 = add_layer(l11, 8, 6, activation_function=tf.nn.sigmoid)
# add output layer 输入值是最后一层隐藏层
prediction = add_layer(l2, 60, 3, activation_function=None)

# 4.定义 loss 表达式 均方误差
loss = tf.losses.mean_squared_error(labels=ys, predictions=prediction)

# 5.定义自适应学习率
# 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率a
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=False)

# 6.选择 optimizer 使 loss 达到最小
# 这一行定义了用什么方式去减少 loss，学习率是 0.1 - 0.0001
# 使用梯度下降算法来最优化损失值
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)


# important step 对所有变量进行初始化
init = tf.global_variables_initializer()
sess = tf.Session()
saver = tf.train.Saver()

# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)
loss_num = []


# 迭代 1000 次学习，sess.run optimizer
for i in range(20000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    rate = sess.run(learning_rate)
    sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
    loss_num.append(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
    if i % 50 == 0:
        # to see the step improvement
        # print(learning_rate[1])
        print(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
    if i % 1000 == 0:
        saver.save(sess, './6_11_4/Model.ckpt', global_step=i)
prediction_value = sess.run(prediction, feed_dict={xs: x_test})
print(prediction_value)
plt.figure(figsize=(10, 5))
plt.plot(loss_num)
plt.show()
