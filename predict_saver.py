import tensorflow as tf
# import cv2

xs = tf.placeholder(tf.float32, [None, 10])
ys = tf.placeholder(tf.float32, [None, 3])
# cnn_path = 'weight_d_cnn/'
saver = tf.train.Saver()
bp_path = 'weight_bp'
sess_bp = tf.Session()
check_bp = tf.train.get_checkpoint_state(bp_path)
init = tf.global_variables_initializer()
if check_bp and check_bp.model_checkpoint_path:
    saver.restore(sess_bp, check_bp.model_checkpoint_path)
else:
    sess_bp.run(init)
# sess_cnn = tf.Session()
# saver_cnn = tf.train.import_meta_graph(cnn_path + 'Model.ckpt-46.meta')
# saver_cnn.restore(sess_cnn, tf.train.latest_checkpoint(cnn_path))
# sess_bp = tf.Session()
# saver_bp = tf.train.import_meta_graph(bp_path + 'Model.ckpt.meta')
# saver_bp.restore(sess_bp, tf.train.latest_checkpoint(bp_path))
# graph = tf.get_default_graph()
# prob_op = graph.get_operations()
# prediction1 = graph.get_tensor_by_name(prob_op)

k = [[0.58085, 0.79549, 0.86366, 0.48164, 0.95886, 0.43736, 0.94252, 1.1407, 1.1891, 0.86911],
     ['0.32632', '1.1322', '0.17469', '0.69638', '0.065115', '0.77295', '0.72768', '1.0234', '0.59715', '0.64469'],
     [0.72456, 0.90015, 1.0057, 0.96418, 1.1634, 1.0471, 0.93388, 0.43748, 1.1786, 0.49177]]

print(sess_bp.run(prediction, feed_dict={xs: k}))
