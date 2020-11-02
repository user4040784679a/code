import tensorflow as tf
import provider
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

index = 100
_, train_label = provider.loadTrainShapeNet16('data')
_, test_label = provider.loadTestShapeNet16('data')
train_data = np.load("./Results_points_minus_dual_every/train_global_feature3d%d.npy"%index)
test_data = np.load("./Results_points_minus_dual_every/test_global_feature3d%d.npy"%index)

batch_label = np.zeros([len(train_label), 16])
for i in range(len(train_label)):
    batch_label[i, train_label[i]] = 1
train_label = np.array(batch_label)

batch_label = np.zeros([len(test_label), 16])
for i in range(len(test_label)):
    batch_label[i, test_label[i]] = 1
test_label = np.array(batch_label)

x = tf.placeholder(tf.float32, [None, 512])
y = tf.placeholder(tf.float32, [None, 16])

net = tf.layers.dense(x, 256, tf.nn.relu)
net = tf.layers.dense(net, 128, tf.nn.relu)
net = tf.layers.dense(net, 16)
pred = tf.nn.softmax(net)

loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))

op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

acc = tf.cast(tf.reduce_mean(tf.cast(
            tf.equal(tf.arg_max(pred, dimension=1),
                     tf.arg_max(y, dimension=1)),
            tf.float32)), tf.float32)

saver = tf.train.Saver(max_to_keep=None)

def train():
    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)

        for i in range(60000):
            idx = np.random.choice(range(len(train_data)), 128)
            train_d = train_data[idx]
            train_l = train_label[idx]
            idx_t = np.random.choice(range(len(test_data)), 128)
            test_d = test_data[idx_t]
            test_l = test_label[idx_t]

            _ = sess.run(op, feed_dict={x: train_d, y: train_l})
            l, a = sess.run([loss, acc], feed_dict={x: train_d, y: train_l})

            print('train', i, ' loss:', l, ' acc:', a)
            lt, at = sess.run([loss, acc], feed_dict={x: test_d, y: test_l})

            print('test ', i, ' loss:', lt, ' acc:', at)

            file_name = 'class_log%d.txt' % index
            with open(file_name, 'a') as f:
                f.write(str(l) + ',' + str(a) + ',' + str(lt) + ',' + str(at) + '\n')

        test_loss = 0
        test_acc = 0
        for i in range(len(test_data)):
            ll, aa = sess.run([loss, acc], feed_dict={x: [test_data[i]], y: [test_label[i]]})
            test_loss += ll
            test_acc += aa
        print('test', ' loss:', test_loss / len(test_data), ' acc:', test_acc / len(test_data))

train()
