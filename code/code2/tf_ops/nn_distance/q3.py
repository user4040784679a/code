import tensorflow as tf
import numpy as np
import time 


tf.enable_eager_execution()

point_num = 2048

def moch(arg):
    pc1, pc2 = arg
        
    # 한 포인트에서 다른 포인트 고려하는 부분
    def modifed_chamfer(p0):
        t = pc2 - p0
        return tf.norm(t, axis=1)

    # point 별 거리 matrix 만든다
    matrix = tf.vectorized_map(modifed_chamfer, pc1)

    max_vec = tf.constant(np.ones(point_num) * 10000, dtype=tf.float32)
    row_vec = tf.expand_dims(max_vec, axis=0)
    col_vec = tf.expand_dims(max_vec, axis=1)
    end_arr = []

    flag = tf.cast(10000, dtype=tf.float32)
    dis_arr = []

    s = time.time()
    while tf.not_equal(tf.reduce_min(matrix), flag):
        for i in range(point_num):
            if i in end_arr:
                continue
            
            t = matrix[i]
            m = tf.argmin(t)
            k = tf.argmin(matrix[:, m])
            
            if tf.equal(tf.cast(i, dtype=tf.int64), k):
                d = tf.reduce_min(t)
                dis_arr.append(d)
                
                matrix = tf.concat([matrix[:i], row_vec, matrix[i+1:]], axis=0)
                matrix = tf.concat([matrix[:, :m], col_vec, matrix[:, m+1:]], axis=1)        
                end_arr.append(i)

    loss = tf.reduce_mean(dis_arr) 

    return loss, 0.

def main(_):
    pc1 = tf.random.normal([4, point_num, 3], dtype=tf.float32)
    pc2 = tf.random.normal([4, point_num, 3], dtype=tf.float32)

    mo_chamfer = tf.reduce_mean(tf.map_fn(moch, (pc1, pc2))[0])
    
    print(mo_chamfer)
    
if __name__ == '__main__':
     tf.app.run()
     


