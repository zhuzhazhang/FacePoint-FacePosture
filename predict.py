import Function
import tensorflow as tf
import cv2
import numpy as np


# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = wx_plus_b
    else:
        outputs = activation_function(wx_plus_b)
    return outputs


# 1. D_CNN模型定义
# 定义D_CNN的变量
cnn_x = tf.placeholder(tf.float32, shape=[None, 39, 39], name='x')
cnn_y = tf.placeholder(tf.float32, shape=[None, 10], name='y')
# 定义D_CNN模型
# -1为缺省值代表这一维度未知 由计算机自己计算
x_image = tf.reshape(cnn_x, [-1, 39, 39, 1])
# 前一层 前一层通道数 卷积核尺寸 卷积核数目
ConvLayer_1, ConvWeights_1 = Function.new_conv_layer(x_image, 1, 4, 20)
ConvLayer_2, ConvWeights_2 = Function.new_conv_layer(ConvLayer_1, 20, 3, 40)
ConvLayer_3, ConvWeights_3 = Function.new_conv_layer(ConvLayer_2, 40, 3, 60)
ConvLayer_4, ConvWeights_4 = Function.new_conv_layer(ConvLayer_3, 60, 2, 80, use_pooling=False)
FlatLayer, FeaturesNum = Function.flatten_layer(ConvLayer_4)
FcLayer1 = Function.new_fc_layer(FlatLayer, FeaturesNum, 120)
FcLayer2 = Function.new_fc_layer(FcLayer1, 120, 10)
Cost = tf.reduce_mean(abs(cnn_y - FcLayer2)/39)
Optimizer = tf.train.AdamOptimizer(1e-4).minimize(Cost)


# 权重加载
checkpoint_d_cnn = './weight_d_cnn/Model.ckpt-46'
checkpoint_bp = '6_11_3/'
saver = tf.train.Saver()
check_d_cnn = tf.train.get_checkpoint_state(checkpoint_d_cnn)
# check_bp = tf.train.get_checkpoint_state(checkpoint_bp)
session_cnn = tf.Session()
# session_bp = tf.Session()
if check_d_cnn and check_d_cnn.model_checkpoint_path:
    saver.restore(session_cnn, check_d_cnn.model_checkpoint_path)
else:
    session_cnn.run(tf.global_variables_initializer())
# if check_bp and check_bp.model_checkpoint_path:
#     saver.restore(session_bp, check_bp.model_checkpoint_path)
# else:
#     session_bp.run(tf.global_variables_initializer())


# 加载图像
images = []
FilePath = './2.jpg'
SrcImage = cv2.imread(FilePath)
cv2.imshow("SrcImage", SrcImage)
cv2.waitKey(1000)
TestImage = cv2.cvtColor(SrcImage, cv2.COLOR_BGR2GRAY)
classfier = cv2.CascadeClassifier("./haarcascade_frontalface_alt2.xml")

face_rect = classfier.detectMultiScale(TestImage, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
if len(face_rect) > 0:  # 大于0则检测到人脸
    for faceRect in face_rect:  # 单独框出每一张人脸
        x, y, w, h = faceRect
        image = cv2.resize(TestImage[int(x + 15):int(x + w - 5), int(y + 15):int(y + h - 5)], (39, 39))
        images.append(image)
        cv2.imshow("image", image)
        cv2.waitKey(1000)


# 开始预测
predict_cnn = session_cnn.run(FcLayer2, feed_dict={cnn_x: images})
# k = [[0.58085, 0.79549, 0.86366, 0.48164, 0.95886, 0.43736, 0.94252, 1.1407, 1.1891, 0.86911],
#      ['0.32632', '1.1322', '0.17469', '0.69638', '0.065115', '0.77295', '0.72768', '1.0234', '0.59715', '0.64469'],
#      [0.72456, 0.90015, 1.0057, 0.96418, 1.1634, 1.0471, 0.93388, 0.43748, 1.1786, 0.49177]]
# predict_bp = session_bp.run(prediction, feed_dict={bp_x: k})
print(predict_cnn)

for p in range(4):
    cv2.circle(images[0], (int(predict_cnn[0][p * 2] * w), int(predict_cnn[0][p * 2 + 1] * w)), 1, (0, 255, 0))
cv2.imshow("test", images[0])
cv2.waitKey(100000)


# print(predict_bp)

