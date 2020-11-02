import tensorflow as tf
import provider
import numpy as np
import os
from sklearn.svm import SVC,LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def get_test_data():
    d0, l0, s0 = provider.loadDataFile_with_seg('./hdf5_data/ply_data_test0.h5')
    d1, l1, s1 = provider.loadDataFile_with_seg('./hdf5_data/ply_data_test1.h5')
    
    return np.concatenate((d0, d1)), np.concatenate((l0, l1)), np.concatenate((s0, s1))

def get_train_data():
    d_arr, l_arr, s_arr = [], [], []
    
    for i in range(6):
        d, l, s = provider.loadDataFile_with_seg('./hdf5_data/ply_data_train{}.h5'.format(i))
        d_arr.append(d)
        l_arr.append(l)
        s_arr.append(s)
    
    d_arr, l_arr, s_arr = np.concatenate(d_arr), np.concatenate(l_arr), np.concatenate(s_arr)
    
    return d_arr, l_arr, s_arr

# train_data = np.load('Results/test_pointae/train_glf.npy')
# test_data = np.load('Results/test_pointae/test_glf.npy')

train_data = np.load('Results/train_kc_enc_pt_dec/train_glf.npy')
test_data = np.load('Results/train_kc_enc_pt_dec/test_glf.npy')

_, train_label, _ = get_train_data()
_, test_label, _ = get_test_data()

train_label = train_label[:len(train_data)].flatten()
test_label = train_label[:len(test_data)].flatten()

print(train_data.shape)
print(train_label.shape)
print(test_data.shape)
print(test_label.shape)

total_acc = 0

clf = LinearSVC()
clf.fit(train_data, train_label)

pred = clf.predict(test_data)

print(pred.shape)
print(test_label.shape)

print(np.mean(pred == test_label))
print(pred[:10])
print(test_label[:10])

# for i in range(100):
#     clf = LinearSVC()
#     clf.fit(train_data, train_label[:,0])
#     pred = clf.predict(test_data)

#     acc = np.mean(pred==test_label[:,0])
#     total_acc += acc
#     print(acc)
# print(total_acc/100.0)
