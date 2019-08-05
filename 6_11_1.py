import tensorflow as tf
# import numpy as np
from read import read_data
from read_test import read_test
# import matplotlib.pyplot as plt


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
# x_test_cnn = read_test("predict.csv")

# 2.定义节点准备接收数据
xs = tf.placeholder(tf.float32, [None, 10])
ys = tf.placeholder(tf.float32, [None, 3])

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs
l1 = add_layer(xs, 10, 60, activation_function=tf.nn.sigmoid)
l2 = add_layer(l1, 60, 60, activation_function=tf.nn.sigmoid)
# add output layer 输入值是最后一层隐藏层
prediction = add_layer(l2, 60, 3, activation_function=None)

# 4.定义 loss 表达式 均方误差
loss = tf.losses.mean_squared_error(labels=ys, predictions=prediction)
# loss2 = tf.reduce_mean(tf.square(ys - prediction))


# 5.定义自适应学习率 定义global_step
global_step = tf.Variable(0, trainable=False)
# 通过指数衰减函数来生成学习率a
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=False)

# 6.选择 optimizer 使 loss 达到最小 这一行定义了用什么方式去减少 loss，学习率是 0.1 - 0.0001
# 使用梯度下降算法来最优化损失值
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

# 加载模型参数
checkpoint_dir = '6_11_4/'
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
sess = tf.Session()
init = tf.global_variables_initializer()
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    sess.run(init)

# for var in tf.trainable_variables():
#     print(var.name, ':', sess.run(var))
# prediction_value = sess.run(prediction, feed_dict={xs: x_test_cnn})
# print(prediction_value)
# 迭代 1000 次学习，sess.run optimizer
# for i in range(20000):
#     # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
#     rate = sess.run(learning_rate)
#     sess.run(train_step, feed_dict={xs: x_train, ys: y_train})
#     loss_num.append(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
#     if i % 50 == 0:
#         # to see the step improvement
#         # print(learning_rate[1])
#         print(sess.run(loss, feed_dict={xs: x_train, ys: y_train}))
# saver.save(sess, './6_11_1/Model.ckpt')
# k = [[0.58085, 0.79549, 0.86366, 0.48164, 0.95886, 0.43736, 0.94252, 1.1407, 1.1891, 0.86911],
#      ['0.32632', '1.1322', '0.17469', '0.69638', '0.065115', '0.77295', '0.72768', '1.0234', '0.59715', '0.64469'],
#      [0.72456, 0.90015, 1.0057, 0.96418, 1.1634, 1.0471, 0.93388, 0.43748, 1.1786, 0.49177],
#      [0.11923277, 0.05952848, 0.05666177, 0.0641628, 0.09006096, 0.07892206, 0.11689555, 0.17079194, 0.02919506,
#       0.04410316]]
#
# k1 = [[0.3331456, 0.2606551, 0.7859482, 0.2874331, 0.6278928, 0.4801588, 0.36429605, 0.7571912, 0.75095284, 0.77413726],
#       [0.2626616, 0.21653539, 0.8074877, 0.2714175, 0.5432312, 0.6116428, 0.2568146, 0.8145631, 0.69864327, 0.8550299],
#       [0.25254062, 0.2807586, 0.7438922, 0.2617124, 0.40658948, 0.4925395, 0.28085315, 0.81681794, 0.74270386,
#        0.80767137]]

# 加载图像
# images = []
# FilePath = './1.jpg'
# SrcImage = cv2.imread(FilePath)
# cv2.imshow("SrcImage", SrcImage)
# cv2.waitKey(1000)
# TestImage = cv2.cvtColor(SrcImage, cv2.COLOR_BGR2GRAY)
# classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")
#
# face_rect = classfier.detectMultiScale(TestImage, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
# if len(face_rect) > 0:  # 大于0则检测到人脸
#     for faceRect in face_rect:  # 单独框出每一张人脸
#         x, y, w, h = faceRect
#         image = cv2.resize(TestImage[int(x + 15):int(x + w - 5), int(y + 15):int(y + h - 5)], (39, 39))
#         images.append(image)
#         cv2.imshow("image", image)
#         cv2.waitKey(10000)
# print(x_test[1990:2000])
# print()
# x_test_1 = [[-0.2204033, -0.039132673371887194, 0.27005124, 0.01888702429428102, 0.0, -0.2867436,
#           -0.12228155, -0.5946132727752685, 0.25024903, -0.5782157965789795]]
# x_test_2 = [[-0.2174222, -0.06346096478805541, 0.28198394, -0.012605566609954821, 0.0, -0.2867436,
#              -0.11717129, -0.5695543476234436, 0.2753611, -0.5436675735603332]]
# x_test_3 = [[-0.32326323, -0.04530312501296996, 0.16987866, -0.021940577853775012, 0.0, -0.2867436,
#              -0.225954, -0.5753233619819641, 0.16683006, -0.5491570421348572]]
# x_test_4 = [[-0.2258814, -0.10190676533088683, 0.24842513, -0.0818519481311798, 0.0,
#              -0.2867436, -0.16610754, -0.597647924054718, 0.20350325, -0.5993387290130615]]
# x_test_5 = [[-0.25143644, 0.004227738748931897, 0.24449575, 0.03904322899475099, 0.0,
#              -0.2867436, -0.14082536, -0.5147416182647705, 0.23975182, -0.4954173751960754]]
# x_test_6 = [[-0.25278804, -0.11554105960235594, 0.20884013, -0.07439463220939635, 0.0,
#              -0.2867436, -0.13256145, -0.6001164504180908, 0.21677655, -0.578468281854248]]
# x_test_7 = [[-0.15109494, -0.057760793794250476, 0.32631058, -0.015588391650772082, 0.0,
#              -0.2867436, -0.13428268, -0.6338350363861084, 0.2353397, -0.6229387112747192]]
# x_test_8 = [[-0.16978794, 0.0017678252567291386, 0.27410972, 0.03033665932312013, 0.0,
#              -0.2867436, -0.09118253, -0.5089955516944885, 0.2609737, -0.506455320943450]]
# x_test_9 = [[-0.3204257, -0.017249871123886096, 0.16571349, -0.010870385993576037, 0.0,
#              -0.2867436, -0.24352047, -0.5115757175575256, 0.12853402, -0.5035242029319763]]
# x_test = [[0.1879, 0.058548, 0.42715, -0.25695, 0, -0.28674, -0.071767, -0.33668, 0.13524, -0.61174]]
# prediction_value = sess.run(prediction, feed_dict={xs: x_test_6})
loss = sess.run(loss, feed_dict={xs: x_train, ys: y_train})

print(loss)
# print(prediction_value)
# print(prediction_value[0:100])
# plt.figure(figsize=(10, 5))
# plt.plot(loss_num)
# plt.show()
