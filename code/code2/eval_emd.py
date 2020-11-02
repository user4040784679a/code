import tensorflow as tf
import numpy as np
import tf_ops.approxmatch.tf_approxmatch as tfappr
from tqdm import tqdm

tf.enable_eager_execution()

name = 'train_chamfer_unified'
data_path = f'./Results/{name}/images/'
batch_size = 64

def eval(xyz1, xyz2):
    loss_arr = []
    
    for i in tqdm(range(xyz1.shape[0] // batch_size)):
        s_idx, e_idx = i * batch_size, (i+1) * batch_size
        
        match = tfappr.approx_match(xyz1[s_idx:e_idx], xyz2[s_idx:e_idx])
        
        mk = np.argmax(match[0].numpy(), 1)
        
        loss = tfappr.match_cost(xyz1[s_idx:e_idx], xyz2[s_idx:e_idx], match).numpy()
        
        loss_arr.append(loss)
    
    loss_arr = np.array(loss_arr)
    loss_arr = loss_arr / 2048
    
    return np.mean(loss_arr), np.std(loss_arr)

gt_points = np.load(data_path + 'shape16_test_data.npy')
pred_points = np.load(data_path + 'shape16_test_encoder3d.npy')
gt_points = gt_points[:pred_points.shape[0]]

print('gt shape', gt_points.shape)
print('pred shape', pred_points.shape)

eval_loss = eval(gt_points, pred_points)
print(eval_loss)
