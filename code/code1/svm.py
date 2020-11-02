import tensorflow as tf
import provider
import numpy as np
import os
from sklearn.svm import SVC,LinearSVC

index = 1300
# _, train_label = provider.loadTrainShapeNet16('data')
# _, test_label = provider.loadTestShapeNet16('data')

_, train_label = provider.loadTrainModel40('data')
_, test_label = provider.loadTestModel40('data')
train_data = np.load("./Results_points_minus_dual_every/model40_train_global_feature3d%d.npy"%index)
test_data = np.load("./Results_points_minus_dual_every/model40_test_global_feature3d%d.npy"%index)


print(train_data.shape)
print(test_data.shape)
print(train_label.shape)
print(test_label.shape)
total_acc = 0
for i in range(100):
    clf = LinearSVC()
    clf.fit(train_data, train_label[:,0])
    pred = clf.predict(test_data)

    acc = np.mean(pred==test_label[:,0])
    total_acc += acc
    print(acc)
print(total_acc/100.0)
# x_data = tf.placeholder(shape=[None, 1024], dtype=tf.float32)
# y_target = tf.placeholder(shape=[None, 40], dtype=tf.float32)

# # 创建变量
# A = tf.Variable(tf.random_normal(shape=[1024, 40]))
# b = tf.Variable(tf.random_normal(shape=[1, 40]))

# # 定义线性模型
# model_output = tf.subtract(tf.matmul(x_data, A), b)

# # Declare vector L2 'norm' function squared
# l2_norm = tf.reduce_sum(tf.square(A))

# # Loss = max(0, 1-pred*actual) + alpha * L2_norm(A)^2
# alpha = tf.constant([0.0])
# classification_term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, y_target))))
# loss = tf.add(classification_term, tf.multiply(alpha, l2_norm))
# op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# acc = tf.cast(tf.reduce_mean(tf.cast(
#             tf.equal(tf.arg_max(model_output, dimension=1),
#                      tf.arg_max(y_target, dimension=1)),
#             tf.float32)), tf.float32)

# saver = tf.train.Saver(max_to_keep=None)

# def train():
#     with tf.Session() as sess:
#         tf.global_variables_initializer().run(session=sess)

#         for i in range(10000000):
#             idx = np.random.choice(range(len(train_data)), 128)
#             train_d = train_data[idx]
#             train_l = train_label[idx]
#             idx_t = np.random.choice(range(len(test_data)), 128)
#             test_d = test_data[idx_t]
#             test_l = test_label[idx_t]

#             _ = sess.run(op, feed_dict={x_data: train_d, y_target: train_l})
#             l, a = sess.run([loss, acc], feed_dict={x_data: train_d, y_target: train_l})

#             print('train', i, ' loss:', l, ' acc:', a)
#             lt, at = sess.run([loss, acc], feed_dict={x_data: test_d, y_target: test_l})

#             print('test ', i, ' loss:', lt, ' acc:', at)

#             file_name = 'svm_log%d.txt' % index
#             with open(file_name, 'a') as f:
#                 f.write(str(l) + ',' + str(a) + ',' + str(lt) + ',' + str(at) + '\n')

#         test_loss = 0
#         test_acc = 0
#         for i in range(len(test_data)):
#             ll, aa = sess.run([loss, acc], feed_dict={x_data: [test_data[i]], y_target: [test_label[i]]})
#             test_loss += ll
#             test_acc += aa
#         print('test', ' loss:', test_loss / len(test_data), ' acc:', test_acc / len(test_data))

# train()
