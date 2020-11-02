import tensorflow as tf
import numpy as np
import os
import utils
import imageio
import glob
import cv2
import provider
import sys
import tf_ops.nn_distance.tf_nndistance as tfnndis
import json
import h5py
from pathlib import Path
from scipy.special import softmax
from tqdm import tqdm
from eval_point_pair import eval_pair
from tensorboardX import SummaryWriter

np.random.seed(0)

LOG_BASE_PATH = './Results'
LOG_BASE_PATH = os.path.join(LOG_BASE_PATH, Path(__file__).stem)
LOG_LOG_PATH = os.path.join(LOG_BASE_PATH, 'log')
LOG_MODEL_PATH = os.path.join(LOG_BASE_PATH, 'model')
LOG_IMAGE_PATH = os.path.join(LOG_BASE_PATH, 'images')
LOG_IMAGE_FULL_PATH = os.path.join(LOG_IMAGE_PATH, 'full')
LOG_IMAGE_labels_PATH = os.path.join(LOG_IMAGE_PATH, 'labels')
LOG_IMAGE_FULL_T_PATH = os.path.join(LOG_IMAGE_PATH, 'full_T')
LOG_IMAGE_labels_T_PATH = os.path.join(LOG_IMAGE_PATH, 'labels_T')

if not os.path.exists(LOG_BASE_PATH):
    os.mkdir(LOG_BASE_PATH)
if not os.path.exists(LOG_LOG_PATH):
    os.mkdir(LOG_LOG_PATH)
if not os.path.exists(LOG_MODEL_PATH):
    os.mkdir(LOG_MODEL_PATH)
if not os.path.exists(LOG_BASE_PATH):
    os.mkdir(LOG_BASE_PATH)
if not os.path.exists(LOG_IMAGE_PATH):
    os.mkdir(LOG_IMAGE_PATH)
if not os.path.exists(LOG_IMAGE_FULL_PATH):
    os.mkdir(LOG_IMAGE_FULL_PATH)
if not os.path.exists(LOG_IMAGE_labels_PATH):
    os.mkdir(LOG_IMAGE_labels_PATH)
if not os.path.exists(LOG_IMAGE_FULL_T_PATH):
    os.mkdir(LOG_IMAGE_FULL_T_PATH)
if not os.path.exists(LOG_IMAGE_labels_T_PATH):
    os.mkdir(LOG_IMAGE_labels_T_PATH)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
hdf5_data_dir = os.path.join(BASE_DIR, './hdf5_data')
TRAINING_FILE_LIST = os.path.join(hdf5_data_dir, 'train_hdf5_file_list.txt')
TESTING_FILE_LIST = os.path.join(hdf5_data_dir, 'val_hdf5_file_list.txt')

NUM_CATEGORIES = 16

all_cats = json.load(open(os.path.join(hdf5_data_dir, 'overallid_to_catid_partid.json'), 'r'))
NUM_PART_CATS = len(all_cats)

def convert_seg_to_one_hot(labels):
    label_one_hot = np.zeros((labels.shape[0],labels.shape[1],NUM_PART_CATS))
    for idx1 in range(labels.shape[0]):
        for idx2 in range(labels.shape[1]):
            label_one_hot[idx1,idx2, labels[idx1,idx2]] = 1
    return label_one_hot

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
    
class AE:
    def build_full_ae(self, trainable=True):

        self.Encoder_3D = []
        self.Decoder_3D = []

        self.T_net = []

        self.input_3D = utils.InputLayer(shape=[self.batch_size, self.points_num, 3, self.points_dim], name='input_3D')
        self.input_labels = utils.InputLayer(shape=[self.batch_size, self.points_num, NUM_PART_CATS, 1], name='input_labels')
        
        #  3D Main Encoder
        for i in range(1, len(self.features_3d)):
            self.Encoder_3D.append(utils.Conv3DLayer(last_layer=self.input_3D if i == 1 else self.Encoder_3D[-1],
                                                     shape=[1,
                                                            self.shapes_3d[i],
                                                            self.features_3d[i - 1],
                                                            self.features_3d[i]],
                                                     strides=[1, 1, 1, 1],
                                                     name='Conv_up_3D_Layer%d_conv' % i,
                                                     padding='VALID',
                                                     activation=None if i == len(
                                                         self.features_3d) - 1 else tf.nn.relu,
                                                     trainable=trainable))

        self.Encoder_3D.append(utils.MaxpoolingLayer(last_layer=self.Encoder_3D[-1], num_point=self.points_num))
        
        self.global_feature = tf.squeeze(self.Encoder_3D[-1].output)
        
        self.Decoder_3D.append(utils.DuplicateLayer(tf.reshape(self.Encoder_3D[-1].output, [self.batch_size, 1, 1, -1]),
                                                    mutiple=self.points_recons, i=1))
        
        self.local_feature = self.Encoder_3D[0]
        self.decoder_out = self.Decoder_3D[-1]
        tt = utils.ConcatLayer(self.local_feature, self.decoder_out)

        for i in range(len(self.fold1_features_conv) - 1, 0, -1):
            self.Decoder_3D.append(utils.Conv3DLayer(last_layer=tt,
                                                     shape=[1,
                                                            self.fold1_shape_conv[i],
                                                            self.fold1_features_conv[i],
                                                            self.fold1_features_conv[i - 1]],
                                                     strides=[1, 1, 1, 1],
                                                     activation=None if i == 1 else tf.nn.relu,
                                                     normalizer=None,
                                                     name='Conv_fold1_3D_Layer%d_conv' % i,
                                                     padding='VALID',
                                                     trainable=trainable))

        # Output node
        self.points = tf.squeeze(self.Decoder_3D[-1].output[:, :, :, :3])
        self.part_label = tf.nn.softmax(self.Decoder_3D[-1].output[:, :, :, 3:])
        
        # label info
        self.part_label_pred = tf.argmax(tf.squeeze(self.part_label), 2)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.part_label_pred,
                                                        tf.argmax(tf.squeeze(self.input_labels.output), 2)), 
                                               tf.float32))
        # Chamfer Distance
        reta,retb,retc,retd = tfnndis.nn_distance(tf.squeeze(self.input_3D.output), self.points)    
        self.loss_idx = retb, retd

        # VAE 1  Optimizer
        with tf.name_scope('Optimizer'):
            # 3D optimizer
            self.loss_3D = tf.reduce_mean(reta)+tf.reduce_mean(retc)
            self.loss_point = tf.reduce_mean(tf.keras.losses.MSE(
                tf.squeeze(self.input_3D.output), self.points))
            
            self.optimizer_3D = tf.train.AdamOptimizer(learning_rate=0.0001,
                                                       name='point_optimizer').minimize(loss=self.loss_point)
            
            # label optimizer
            self.loss_labels = tf.reduce_mean(tf.keras.losses.MSE(tf.squeeze(self.input_labels.output), 
                                                                  tf.squeeze(self.part_label)))
            
            self.optimizer_labels = tf.train.AdamOptimizer(learning_rate=self.lr,
                                                       beta1=self.beta,
                                                       name='Adamvae2d').minimize(loss=self.loss_labels)
        
            # Summary
            self.summary.append(tf.summary.scalar(name='loss_3D', tensor=self.loss_3D))
            self.summary.append(tf.summary.scalar(name='loss_labels', tensor=self.loss_labels))
            self.summary.append(tf.summary.scalar(name='label_accuracy', tensor=self.accuracy))
            

    def __init__(self):
        self.model = []
        self.batch_size = 16
        self.img_width = 128
        self.img_height = 128
        self.points_dim = 1
        self.points_num = 2048

        self.class_dim = 16
        self.labels_dim = NUM_PART_CATS
        self.labels_num = 2048

        self.points_recons = 2048
        self.skip = 1

        self.features_3d = [1, 64, 128, 1024]
        self.shapes_3d = [0, 3, 1, 1]

        self.features_labels = [1, 64, 128, 1024]
        self.shapes_labels = [0, self.labels_dim, 1, 1]
        
        self.fold1_features_conv = [53, 1088, 1088, 1088, 1088, 1088]
        self.fold1_shape_conv = [1, 1, 1, 1, 1, 1]

        self.labels_inverse_features = [self.labels_dim, 512, 512, 512, 512, 1024]
        self.labels_inverse_shape = [1, 1, 1, 1, 1, 1]

        self.strides = np.repeat([2], len(self.features_labels))
        self.padding = np.append(np.repeat("SAME", len(self.features_labels) - 1), "VALID")
        self.summary = []
        self.loss = None
        self.optimizer = None
        self.lr = 0.000125
        self.beta = 0.5

        self.build_full_ae()
        self.summary_groubs = tf.summary.merge(self.summary, name='loss')

        self.saver = tf.train.Saver(max_to_keep=200)
        self.session = tf.Session()
        self.train_writer = tf.summary.FileWriter(LOG_LOG_PATH + '/training', self.session.graph)
        self.test_writer = tf.summary.FileWriter(LOG_LOG_PATH + '/testing', self.session.graph)
        tf.global_variables_initializer().run(session=self.session)
        
        self.pair_writer = SummaryWriter(os.path.join(LOG_BASE_PATH, 'pair'))


    # default 2000
    def train_ae(self, n_epochs=160):

        train_file_list = provider.getDataFiles(TRAINING_FILE_LIST)
        num_train_file = len(train_file_list)

        test_data, test_label, test_seg = provider.loadPartTest('./hdf5_data')
        test_data = np.reshape(test_data, [-1, 2048, 3])
        test_seg = np.reshape(test_seg, [-1, 2048])
        # test_seg = convert_seg_to_one_hot(test_seg)
        test_label = tf.keras.utils.to_categorical(test_label, self.class_dim)
        
        print(test_data.shape, test_seg.shape, test_label.shape)

        index = 0

        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))
        self.saver = tf.train.Saver(max_to_keep=200)
        
        print("start~~~~~~~")
        # self.load_model(LOG_MODEL_PATH + '/fine_tune.ckpt')
        
        for epoch in range(index, n_epochs):

            train_file_idx = np.arange(0, len(train_file_list))
            
            for i in range(num_train_file):
                cur_train_filename = os.path.join(hdf5_data_dir, train_file_list[train_file_idx[i]])

                cur_data, cur_labels, cur_seg = provider.loadDataFile_with_seg(cur_train_filename)
                
                cur_data, cur_labels, order = provider.shuffle_data(cur_data, np.squeeze(cur_labels))
                cur_seg = cur_seg[order, ...]
                
                cur_labels = tf.keras.utils.to_categorical(cur_labels, self.class_dim)
                
                num_data = len(cur_labels)
                num_batch = num_data // self.batch_size

                loss_3d_arr, loss_label_arr, acc_arr = [], [], []
                
#                 for iteration in range(num_batch):
                for iteration in tqdm(range(num_batch)):
                    s_idx, e_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
                    Batch_Domain_3d = cur_data[s_idx:e_idx]
                    Batch_Domain_labels = cur_seg[s_idx:e_idx]
                    Batch_class_label = cur_labels[s_idx:e_idx]
                    
                    Batch_Domain_labels = convert_seg_to_one_hot(Batch_Domain_labels)
                    Batch_Domain_3d = np.expand_dims(Batch_Domain_3d, axis=-1)
                    Batch_Domain_labels = np.expand_dims(Batch_Domain_labels, axis=-1)

                    feed_dict_vae = {self.input_3D.output: Batch_Domain_3d}
                                    
                    l_3d, ret, _ = self.session.run([self.loss_point, self.loss_idx, self.optimizer_3D], 
                                                    feed_dict=feed_dict_vae)
                    loss_3d_arr.append(l_3d)
                    
                    target_labels = Batch_Domain_labels
                    
                    target_feed = {
                        self.input_3D.output: Batch_Domain_3d,
                        self.input_labels.output: target_labels
                    }
                    
                    l_label, acc, _ = self.session.run([self.loss_labels, self.accuracy, self.optimizer_labels], 
                                                       feed_dict=target_feed)
                    loss_label_arr.append(l_label)
                    acc_arr.append(acc)
                    
                print("\r  %d %d-%d / 3d %f loss %f acc %f" % (epoch, iteration, i, np.mean(loss_3d_arr),
                                                               np.mean(loss_label_arr), np.mean(acc_arr)))
            
            ##### validation
            
            test_idx = np.random.choice(len(test_data), self.batch_size)
            
            num_data = len(test_data)
            num_batch = num_data // self.batch_size
            
            val_loss_arr = []
            val_acc_arr = []
            val_pair_arr = []
            
            for iteration in tqdm(range(num_batch)):
                s_idx, e_idx = iteration * self.batch_size, (iteration + 1) * self.batch_size
            
                Batch_Domain_3dT = test_data[s_idx:e_idx]
                Batch_Domain_labelsT = test_seg[s_idx:e_idx]
                
                Batch_Domain_labelsT = convert_seg_to_one_hot(Batch_Domain_labelsT)
                Batch_Domain_3dT = np.expand_dims(Batch_Domain_3dT, axis=-1)
                Batch_Domain_labelsT = np.expand_dims(Batch_Domain_labelsT, axis=-1)
                
                ret = self.session.run(self.loss_idx, feed_dict={self.input_3D.output: Batch_Domain_3dT})
                retb, retd = ret
                
                val_pair_arr.append(eval_pair(retb))
                
                target_labelsT = Batch_Domain_labelsT
                
                sample_3dT, sample_loss, summary_test, test_acc = self.session.run(
                    [self.points, self.loss_point, self.summary_groubs, self.accuracy],
                    feed_dict={
                        self.input_3D.output: Batch_Domain_3dT,
                        self.input_labels.output: target_labelsT})
                
                val_acc_arr.append(test_acc)
                val_loss_arr.append(sample_loss)
            
            #
            val_pair = np.mean(val_pair_arr)
            val_3d_loss = np.mean(val_loss_arr)
            val_acc = np.mean(val_acc_arr)
            
            self.pair_writer.add_scalar('pair', val_pair, epoch)
            
            print(f'Epoch {epoch} Val 3d loss: {val_3d_loss}, pair: {val_pair} acc: {val_acc}')
            
            # save log and img
            sample_3d = self.session.run(self.points, feed_dict=feed_dict_vae)
            self.test_writer.add_summary(summary_test, epoch)
            self.test_writer.flush()

            try:
                sample_3d_img = np.array([utils.point_cloud_one_views(p)[0] for p in np.squeeze(sample_3d)])
                sample_3dT_img = np.array([utils.point_cloud_one_views(p)[0] for p in np.squeeze(sample_3dT)])

                GT_3d_img = np.array([utils.point_cloud_one_views(p)[0] for p in np.squeeze(Batch_Domain_3d)])
                GT_3dT_img = np.array([utils.point_cloud_one_views(p)[0] for p in np.squeeze(Batch_Domain_3dT)])

                for i in range(len(sample_3d)):
                    imageio.imsave(LOG_IMAGE_FULL_PATH + "/sample_%04d.jpg" % i, sample_3d_img[i])
                    imageio.imsave(LOG_IMAGE_FULL_PATH + "/GT_%04d.jpg" % i, GT_3d_img[i])
                    imageio.imsave(LOG_IMAGE_FULL_T_PATH + "/sample_%04d.jpg" % i, sample_3dT_img[i])
                    imageio.imsave(LOG_IMAGE_FULL_T_PATH + "/GT_%04d.jpg" % i, GT_3dT_img[i])
            except:
                pass
            
            summary_training = self.session.run(self.summary_groubs, feed_dict=target_feed)

            self.train_writer.add_summary(summary_training, epoch)
            self.train_writer.flush()
            self.saver.save(self.session, save_path= LOG_MODEL_PATH + '/fine_tune.ckpt')
            
        self.saver.save(self.session, save_path=LOG_MODEL_PATH + '/last_model.ckpt')

    def load_model(self, model):
        self.saver.restore(self.session, model)

    def get_glf(self, data, label):
        iter_cnt = len(data) // self.batch_size + 1
        glf_arr = []
        
        for idx in tqdm(range(iter_cnt)):
            s_idx, e_idx = idx * self.batch_size, (idx+1) * self.batch_size
            Batch_Domain_3dT = data[s_idx:e_idx]
            Batch_Domain_3dT = np.expand_dims(Batch_Domain_3dT, axis=-1)
            
            if len(Batch_Domain_3dT) != 16:
                continue
            
            glf = self.session.run(self.global_feature, feed_dict={
                self.input_3D.output: Batch_Domain_3dT
            })
            
            glf_arr.append(glf)
            
        glf_arr = np.concatenate(glf_arr)
        np.save(os.path.join(LOG_BASE_PATH, label + '_glf.npy'), glf_arr)
        
    def collect_global_feature(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))
        self.saver = tf.train.Saver(max_to_keep=200)

        self.load_model('./Result_0806/train_pointnet_ae/model/fine_tune.ckpt')
        
        train_data, train_label, train_seg = get_train_data()        
        train_data = np.reshape(train_data, [-1, 2048, 3])
        self.get_glf(train_data, 'train')
                
        test_data, test_label, test_seg = get_test_data()
        test_data = np.reshape(test_data, [-1, 2048, 3])
        
        self.get_glf(test_data, 'test')
        
    # save Point Cloud Output and segmentation label
    # npy data can be displayed using render program
    def collect_result(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))
        self.saver = tf.train.Saver(max_to_keep=200)

        self.load_model(LOG_MODEL_PATH + '/fine_tune.ckpt')

        test_data, test_label, test_seg = get_test_data()
        
        test_data = np.load('test_data/test_data_45_0_0.npy')
        
        print('-----')
        print(test_data.shape, test_label.shape)
        
        test_data = np.reshape(test_data, [-1, 2048, 3])
        test_seg = np.reshape(test_seg, [-1, 2048])
        test_label = tf.keras.utils.to_categorical(test_label, self.class_dim)
        
        np.save(LOG_IMAGE_PATH +"/shape16_test_data.npy",  test_data)
        np.save(LOG_IMAGE_PATH +"/shape16_test_seg.npy",  test_seg)

        testing3d_copy = test_data
        testinglabels_copy = test_seg
        testing3d_copy = np.expand_dims(testing3d_copy, axis=-1)
        # testinglabels_copy = np.expand_dims(testinglabels_copy, axis=-1)

        encoder3dT = []
        labelsT = []
        loss_arr_3d, loss_arr_label = [], []

        for i in range(0, len(test_data), self.batch_size):
            sys.stdout.write("\r test collecting %d / %d" % (int(i / self.batch_size), int(len(test_data) / self.batch_size)))
            sys.stdout.flush()
            
            Batch_Domain_3dT = testing3d_copy[i:i+self.batch_size]
            Batch_Domain_labelsT = testinglabels_copy[i:i+self.batch_size]
            # Batch_Domain_2d = training2d_copy[i:i+32]
            Batch_Domain_labelsT = convert_seg_to_one_hot(Batch_Domain_labelsT)
            Batch_class_labelT = test_label[i:i+self.batch_size]

            # Batch_Domain_3dT = np.expand_dims(Batch_Domain_3dT, axis=-1)
            Batch_Domain_labelsT = np.expand_dims(Batch_Domain_labelsT, axis=-1)

            # Since the fixed batch size is used, the batch size must always be constant.
            if Batch_Domain_3dT.shape[0] != 16:
                continue
            
            target_labelsT = Batch_Domain_labelsT
            
            sample_3dT, sample_labelsT, test_3d_loss, test_label_loss = self.session.run(
                [self.points, self.part_label, self.loss_3D, self.loss_labels],
                feed_dict={
                    self.input_3D.output: Batch_Domain_3dT,
                    self.input_labels.output: target_labelsT})
            
            encoder3dT.append(sample_3dT)
            labelsT.append(sample_labelsT)
            loss_arr_3d.append(test_3d_loss)
            loss_arr_label.append(test_label_loss)

        encoder3dT = np.reshape(np.array(encoder3dT), [-1, self.points_num, 3])
        labelsT = np.reshape(np.array(labelsT), [-1, self.points_num, NUM_PART_CATS])

        encoder3dT = encoder3dT[:len(test_data)]

        labelsT = labelsT[:len(test_data)]
        labelsT = np.argmax(softmax(labelsT, axis=2), axis=2)

        print()
        print(f'3d loss {np.mean(loss_arr_3d)}')
        print(f'3d std {np.std(loss_arr_3d)}')
        print(f'label loss {np.mean(loss_arr_label)}')
        print(labelsT.shape)
        print(labelsT[0][:10])
        
        np.save(LOG_IMAGE_PATH +"/shape16_test_encoder3d.npy", encoder3dT)
        np.save(LOG_IMAGE_PATH +"/shape16_test_encoder_labels.npy", labelsT)

    # Function to store data to calculate accuracy and miou
    def collect_acc(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.session.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]
        if len(not_initialized_vars):
            self.session.run(tf.variables_initializer(not_initialized_vars))
        self.saver = tf.train.Saver(max_to_keep=200)

        self.load_model(LOG_MODEL_PATH + '/fine_tune.ckpt')
        
        f = h5py.File(LOG_BASE_PATH + '/segmentation_result.hdf5', 'w')

        test_data, test_label, test_seg = get_test_data()
        
        test_data = np.load('test_data/test_data_45_0_0.npy')
        print('-----')
        print(test_data.shape, test_label.shape)
        
        test_data = np.reshape(test_data, [-1, 2048, 3])
        test_seg = np.reshape(test_seg, [-1, 2048])
        test_label2 = tf.keras.utils.to_categorical(test_label, self.class_dim)
        
        sample_arr = []
        gt_arr = []
        class_arr = []
        
        # sess run
        for i in range(0, len(test_data), self.batch_size):
            sys.stdout.write("\r test collecting %d / %d" % (int(i / self.batch_size), int(len(test_data) / self.batch_size)))
            sys.stdout.flush()

            Batch_Domain_3dT = test_data[i:i+self.batch_size]
            Batch_Domain_labelsT = test_seg[i:i+self.batch_size]
            Batch_class_label = test_label2[i:i+self.batch_size]
            batch_label = test_label[i:i+self.batch_size]
            
            if Batch_Domain_3dT.shape[0] != 16:
                continue
            
            Batch_Domain_labelsT = convert_seg_to_one_hot(Batch_Domain_labelsT)

            Batch_Domain_3dT = np.expand_dims(Batch_Domain_3dT, axis=-1)
            Batch_Domain_labelsT = np.expand_dims(Batch_Domain_labelsT, axis=-1)
            
            target_labelsT = Batch_Domain_labelsT
            
            sample_labelsT = self.session.run(self.part_label_pred, 
                                              feed_dict={
                                                  self.input_3D.output: Batch_Domain_3dT,
                                                  self.input_labels.output: target_labelsT
                                              })
            
            for idx in range(len(sample_labelsT)):
                tt = sample_labelsT[idx]
                ll = np.argmax(target_labelsT[idx].reshape(2048, 50), 1)
                
                sample_arr.append(tt)
                gt_arr.append(ll)
                class_arr.append(batch_label[idx][0])
                
        f['sample'] = sample_arr
        f['gt'] = gt_arr
        f['class'] = class_arr
        f.close()
    
def train_AE():

    # provider.loadPartTest('./hdf5_data')
    ae = AE()
#     ae.train_ae()
    ae.collect_result()
#     ae.collect_acc()

    print("complete!")


# train_3d, train_label = provider.loadTrainModel40('data')
# test_3d, test_label = provider.loadTestModel40('data')
# np.save("./Results_points_minus_dual_every/model_train_3d.npy", train_3d)
# np.save("./Results_points_minus_dual_every/model_test_3d.npy", test_3d)

# cut_test()

train_AE()