import numpy as np
import h5py
import time
from pprint import pprint
from sklearn.metrics import confusion_matrix  

label_map = [
    'Airplane', 
    'Bag', 
    'Cap', 
    'Car', 
    'Chair', 
    'Earphone', 
    'Guitar', 
    'Knife', 
    'Lamp', 
    'Laptop', 
    'Motorbike', 
    'Mug', 
    'Pistol', 
    'Rocket', 
    'Skateboard', 
    'Table', 
]

def compute_iou(y_pred, y_true, c):
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    
    pred_labels = {x: {'true':0, 'false': 0} for x in list(set(list(y_pred) + list(y_true)))}
    
    for p, t in zip(y_pred, y_true):
        if p == t:
            pred_labels[p]['true'] += 1
        else:
            pred_labels[p]['false'] += 1
            pred_labels[t]['false'] += 1
            
    del_key_arr = []
    
    for k, v in pred_labels.items():      
        if pred_labels[k]['true'] < 80:
            del_key_arr.append(k)
    
    for k in del_key_arr:
        del pred_labels[k]
    
    # print(pred_labels)
    
    iou_arr = []
     
    for v in pred_labels.values():
        iou = v['true'] / (v['true'] + v['false'])
        iou_arr.append(iou)    
        
    # print(np.mean(iou_arr), np.mean(y_pred == y_true))
#     time.sleep(1)
    return np.mean(iou_arr)
            
def main():
    # sample, gt, class
    f = h5py.File('./Results/train_modi_20/segmentation_result.hdf5', 'r')
#     f = h5py.File('./Results/train_cd_mean2/segmentation_result.hdf5', 'r')
    sample_arr = np.array(f['sample'])
    gt_arr = np.array(f['gt'])
    class_arr = np.array(f['class'])
    
    dd = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0}
    idx_arr = []
    miou_arr = []
    acc_arr = []
    
    class_dict = {x: [] for x in range(16)}
    
    for idx in range(len(sample_arr)):
        s, g, c = sample_arr[idx], gt_arr[idx], class_arr[idx]
        k = int(np.mean(s == g) * 10)
        
        if k < 1:
            continue
        
        miou = compute_iou(s, g, c)
        
        miou_arr.append(miou)
        acc = np.mean(s == g)
        acc_arr.append(acc)
        
        class_dict[c].append([miou, acc])
        
#     for k, v in class_dict.items():
#         v = np.array(v)
# #         print(v)
#         print(f'class {label_map[k]} len {len(v)} miou mean {np.mean(v[:, 0])}, acc mean {np.mean(v[:, 1])}')
        
        
    print('miou mean ', np.mean(miou_arr))
    print('miou std ', np.std(miou_arr))
    print('acc mean ', np.mean(acc_arr))
    
if __name__ == '__main__':
    main()