import tensorflow as tf
from tf_ops.nn_distance.tf_nndistance import nn_distance
import numpy as np


class ChamferDistance:
    def __init__(self, point1_num, point2_num):
        with tf.variable_scope('chamfer'):
            self.point_input1 = tf.placeholder(tf.float32, [None, point1_num, 3])
            self.point_input2 = tf.placeholder(tf.float32, [None, point2_num, 3])
            
            reta, retb, retc, retd = nn_distance(self.point_input1, self.point_input2)
            
            self.idx_1_to_2 = retb
            self.idx_2_to_1 = retd
            
            self.dist1 = tf.reduce_sum(reta)
            self.dist2 = tf.reduce_sum(retc)        
            self.chamfer_distance = self.dist1 + self.dist2

    def loss(self, point1, point2, sess):
        chamfer_loss, dist1, dist2 = sess.run([self.chamfer_distance,
                                               self.dist1, self.dist2], 
                                feed_dict={self.point_input1: point1, 
                                           self.point_input2: point2})
        
        return chamfer_loss, dist1, dist2
    
    def get_idx(self, point1, point2, sess):
        idx_1_to_2, idx_2_to_1 = sess.run([self.idx_1_to_2, self.idx_2_to_1],
                                          feed_dict={self.point_input1: point1,
                                                     self.point_input2: point2})
        
        return idx_1_to_2, idx_2_to_1

    
def cnt_dis_info(d):
    unique, counts = np.unique(d, return_counts=True)
    
    one_count = np.sum(counts == 1)
    
    return one_count
    
def eval_pair(idx_data):
    result_arr = []

    for d in idx_data:
        result = cnt_dis_info(d)
        result_arr.append(result)
    
    return np.mean(result_arr)

if __name__=='__main__':
    np.random.seed(100)
    
#     xyz1 = np.load('Results/seg_result/shape16_test_encoder3d.npy')
#     xyz2 = np.load('Results/seg_result/shape16_test_data.npy')[:2864]

#     xyz1 = np.load('Results/train_modi_5/images/shape16_test_encoder3d.npy')
#     xyz2 = np.load('Results/train_modi_5/images/shape16_test_data.npy')[:2864]

#     xyz1 = np.load('Results/train_modi_10_multi/images/shape16_test_encoder3d.npy')
#     xyz2 = np.load('Results/train_modi_10_multi/images/shape16_test_data.npy')[:2832]
    
#     xyz1 = np.load('Results/train_modi_20_multi/modi20/shape16_test_encoder3d.npy')
#     xyz2 = np.load('Results/train_modi_20_multi/modi20/shape16_test_data.npy')[:2864]
    
    with tf.Session() as sess:
        idx_from1, idx_from2 = chamfer.get_idx(xyz1, xyz2, sess)  
         
        eval_pair(idx_from1)
