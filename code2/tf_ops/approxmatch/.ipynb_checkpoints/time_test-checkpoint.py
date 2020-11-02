""" Approxmiate algorithm for computing the Earch Mover's Distance.

Original author: Haoqiang Fan
Modified by Charles R. Qi
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
approxmatch_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_approxmatch_so.so'))
def approx_match(xyz1,xyz2):
    '''
input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
returns:
    match : batch_size * #query_points * #dataset_points
    '''
    return approxmatch_module.approx_match(xyz1,xyz2)
ops.NoGradient('ApproxMatch')
#@tf.RegisterShape('ApproxMatch')
#def _approx_match_shape(op):
#	shape1=op.inputs[0].get_shape().with_rank(3)
#	shape2=op.inputs[1].get_shape().with_rank(3)
#	return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[1]])]

def match_cost(xyz1,xyz2,match):
    '''
input:
    xyz1 : batch_size * #dataset_points * 3
    xyz2 : batch_size * #query_points * 3
    match : batch_size * #query_points * #dataset_points
returns:
    cost : batch_size
    '''
    return approxmatch_module.match_cost(xyz1,xyz2,match)
#@tf.RegisterShape('MatchCost')
#def _match_cost_shape(op):
#	shape1=op.inputs[0].get_shape().with_rank(3)
#	shape2=op.inputs[1].get_shape().with_rank(3)
#	shape3=op.inputs[2].get_shape().with_rank(3)
#	return [tf.TensorShape([shape1.dims[0]])]
@tf.RegisterGradient('MatchCost')
def _match_cost_grad(op,grad_cost):
    xyz1=op.inputs[0]
    xyz2=op.inputs[1]
    match=op.inputs[2]
    grad_1,grad_2=approxmatch_module.match_cost_grad(xyz1,xyz2,match)
    return [grad_1*tf.expand_dims(tf.expand_dims(grad_cost,1),2),grad_2*tf.expand_dims(tf.expand_dims(grad_cost,1),2),None]

if __name__=='__main__':
    alpha=0.5
    beta=2.0
    import numpy as np
    import math
    import random
    import cv2


    npoint=2048

    pt_in=tf.placeholder(tf.float32,shape=(32,npoint,3))
    mypoints=tf.Variable(np.random.randn(32,npoint,3).astype('float32'))
    match=approx_match(pt_in,mypoints)
    loss=tf.reduce_sum(match_cost(pt_in,mypoints,match))
    
    optimizer=tf.train.AdamOptimizer(1e-4).minimize(loss)
    t_points = np.random.randn(32, 2048, 3).astype('float32')
    
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
        meanloss=0
        meantrueloss=0
        time_arr = []
        
        for i in range(100):
            s = time.time()
            trainloss=sess.run([loss],feed_dict={pt_in: t_points})
            time_arr.append(time.time() - s)
        
        print(np.mean(time_arr))
        