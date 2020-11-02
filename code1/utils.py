import tensorflow as tf
from functools import reduce
from tensorflow.contrib.slim import fully_connected as fc
import tensorflow.contrib.slim as slim
from eulerangles import euler2mat
import numpy as np
import cv2


class TransformLayer:
    def __init__(self,
                 last_layer,
                 name,
                 operator,
                 shape,
                 reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            self.weight = tf.get_variable(name='weights',
                                          initializer=tf.tile(tf.constant([[1, 0, 0, 0, 1, 0, 0, 0, 1]], dtype=tf.float32), [shape[0], 1]))
            self.t = tf.reshape(tf.add(self.weight, operator.output), [-1, 3, 3])
            self.output = tf.expand_dims(tf.matmul(tf.squeeze(last_layer.output), self.t), -1)

class GridLayer:
    def __init__(self,
                 last_layer,
                 batch_size,
                 meshgrid,
                 name,
                 trainable=True,
                 reuse=False):
        with tf.variable_scope(name, reuse=reuse):
            ret = np.meshgrid(*[np.linspace(it[0], it[1], num=it[2]) for it in meshgrid])
            ndim = len(meshgrid)
            grid = np.zeros((np.prod([it[2] for it in meshgrid]), ndim), dtype=np.float32)  # MxD
            for d in range(ndim):
                grid[:, d] = np.reshape(ret[d], -1)
            g = np.repeat(grid[np.newaxis, ...], repeats=batch_size, axis=0)

            self.grid = tf.get_variable(name='weights',
                                        initializer=tf.constant(g),
                                        trainable=False)
            self.output = tf.concat([tf.squeeze(last_layer.output), self.grid], axis=2)
            self.output = tf.expand_dims(self.output, 2)

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def mse(y1, y2):
    assert y1.shape.as_list() == y2.shape.as_list()
    return tf.reduce_sum(tf.pow(y1 - y2, 2)) / (reduce(lambda x, y: x * y, y1.shape.as_list()))

def mse_2(y1, y2):
    assert y1.shape.as_list() == y2.shape.as_list()
    return tf.reduce_sum(tf.pow(y1 - y2, 2), axis=1) / (reduce(lambda x, y: x * y, y1.shape.as_list()))


def mse0(y):
    return tf.reduce_mean(tf.pow(y, 2))


def temp_sigmoid(x, temp=10.0):
    return 1.0 / (1 + tf.exp(-tf.multiply(x, temp)))


def chamber_loss(x, y):
    x_size = x.shape
    y_size = y.shape
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = tf.expand_dims(x, axis=1)  # x = batch,1,2025,3
    y = tf.expand_dims(y, axis=2)  # y = batch,2048,1,3

    x = tf.tile(x, [1, y_size[1], 1, 1])
    y = tf.tile(y, [1, 1, x_size[1], 1])

    x_y = x - y
    x_y = tf.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = tf.reduce_sum(x_y, axis=3, keep_dims=True)  # x_y = batch,2048,2025,1
    x_y = tf.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y_row = tf.reduce_min(x_y, axis=1, keep_dims=True)  # x_y_row = batch,1,2025
    x_y_col = tf.reduce_min(x_y, axis=2, keep_dims=True)  # x_y_col = batch,2048,1

    x_y_row = tf.reduce_mean(x_y_row, 2, keep_dims=True)  # x_y_row = batch,1,1
    x_y_col = tf.reduce_mean(x_y_col, 1, keep_dims=True)  # batch,1,1
    x_y_row_col = tf.concat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance = tf.reduce_max(x_y_row_col, axis=2, keep_dims=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = tf.reduce_mean(chamfer_distance)
    return chamfer_distance

def chamber_loss_2(x, y):
    x_size = x.shape
    y_size = y.shape
    assert (x_size[0] == y_size[0])
    assert (x_size[2] == y_size[2])
    x = tf.expand_dims(x, axis=1)  # x = batch,1,2025,3
    y = tf.expand_dims(y, axis=2)  # y = batch,2048,1,3

    x = tf.tile(x, [1, y_size[1], 1, 1])
    y = tf.tile(y, [1, 1, x_size[1], 1])

    x_y = x - y
    x_y = tf.pow(x_y, 2)  # x_y = batch,2048,2025,3
    x_y = tf.reduce_sum(x_y, axis=3, keep_dims=True)  # x_y = batch,2048,2025,1
    x_y = tf.squeeze(x_y, 3)  # x_y = batch,2048,2025
    x_y_row = tf.reduce_min(x_y, axis=1, keep_dims=True)  # x_y_row = batch,1,2025
    x_y_col = tf.reduce_min(x_y, axis=2, keep_dims=True)  # x_y_col = batch,2048,1

    x_y_row = tf.reduce_mean(x_y_row, 2, keep_dims=True)  # x_y_row = batch,1,1
    x_y_col = tf.reduce_mean(x_y_col, 1, keep_dims=True)  # batch,1,1
    x_y_row_col = tf.concat((x_y_row, x_y_col), 2)  # batch,1,2
    chamfer_distance = tf.reduce_max(x_y_row_col, axis=2, keep_dims=True)  # batch,1,1
    # chamfer_distance = torch.reshape(chamfer_distance,(x_size[0],-1))  #batch,1
    # chamfer_distance = torch.squeeze(chamfer_distance,1)    # batch
    chamfer_distance = tf.reduce_mean(chamfer_distance, axis=1)
    return chamfer_distance

class Tensor:
    def __init__(self, output):
        self.output = output

class InputLayer:
    def __init__(self,
                 shape,
                 name,
                 dtype=tf.float32):
        self.output = tf.placeholder(dtype=dtype, shape=shape, name=name)

class MaxpoolingLayer:
    def __init__(self, last_layer, num_point):
        self.output = tf.nn.max_pool(last_layer.output, [1, num_point, 1, 1], strides=[1, 2, 2, 1], padding='VALID')
        self.output = tf.reshape(self.output, [self.output.shape[0], 1, 1, -1])

class DuplicateLayer:
    def __init__(self, last_layer, mutiple, i=0):
        if i == 0:
            self.output = tf.tile(last_layer.output, [1, mutiple, 1, 1])
        else:
            self.output = tf.tile(last_layer, [1, mutiple, 1, 1])


class ConcatLayer:
    def __init__(self, last_layer, concat_Layer):
        self.output = tf.concat((last_layer.output, concat_Layer.output), axis=-1)

class MinusLayer:
    def __init__(self,
                 last_layer,
                 shape,
                 name,
                 activation_fc=tf.nn.relu,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False,
                 i=0
                 ):
        self.conv3Vars = {}
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            # self.add_weight = tf.Variable(tf.truncated_normal(shape=last_layer.output.shape.as_list()))
            self.add_weight = tf.get_variable(name='weights', shape=shape)
            
            if i == 0:
                self.output = tf.add(last_layer.output, self.add_weight)
            else:
                self.output = tf.add(last_layer, self.add_weight)
                
            if activation_fc:
                self.output = activation_fc(self.output)
                
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
        if not reuse:
            self.conv3Vars.update({name: self.trainable_variables})


class RandomLayer:
    def __init__(self, shape):
        self.output = tf.Variable(tf.truncated_normal(shape=shape))

class LocalConvLayer:
    def __init__(self,
                 last_layer,
                 shape,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="VALID",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False):
        self.conv3Vars = {}
        self.vars = {}
        self.optimizers = {}
        self.lr = 0.0001
        self.beta = 0.5
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.padding = padding
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            self.weights = tf.get_variable(name='weights', shape=shape, initializer=initializer, dtype=tf.float32)
            self.conv = tf.matmul(self.weights, tf.transpose(last_layer.output, perm=[0, 1, 3, 2]))
            if normalizer is None:
                    self.batch_norm = self.conv
            else:
                    self.batch_norm = normalizer(self.conv, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
        if not reuse:
            self.conv3Vars.update({name: self.trainable_variables})


class Conv3DLayer:
    def __init__(self,
                 last_layer,
                 shape,
                 strides,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        self.conv3Vars = {}
        self.vars = {}
        self.optimizers = {}
        self.lr = 0.0001
        self.beta = 0.5
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.strides = strides
            self.padding = padding
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            self.weights = tf.get_variable(name='weights', shape=shape, initializer=initializer, dtype=tf.float32)
            self.conv3d = tf.nn.conv2d(last_layer.output, self.weights, strides=strides, padding=padding)
            
            if normalizer is None:
                self.batch_norm = self.conv3d
            else:
                self.batch_norm = normalizer(self.conv3d, is_training=trainable)
            
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
                
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
            
        if not reuse:
            self.conv3Vars.update({name: self.trainable_variables})

class Deconv3DLayer:
    def __init__(self,
                 deco,
                 i,
                 last_layer,
                 shape,
                 out_shape,
                 strides,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 padding="SAME",
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        self.deconv3Vars = {}
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.deco=deco
            self.last_layer=last_layer
            self.strides = strides
            self.padding = padding
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            self.out_shape = out_shape
            self.weights = tf.get_variable('weights', shape=shape, initializer=initializer)
            if len(last_layer.output.shape.as_list())<4:
                last_layer.output=tf.reshape(last_layer.output,[last_layer.output.shape.as_list()[0],shape[0],shape[1],shape[3]])

            if self.deco==1:
                if i==6:
                    self.deconv3d = tf.nn.conv2d_transpose(last_layer.output, self.weights, strides=strides, padding=padding,output_shape=out_shape)
                else:
                    self.deconv3d = tf.nn.conv2d_transpose(last_layer.output, self.weights, strides=strides,padding=padding, output_shape=out_shape)
            else:
                self.deconv3d = tf.nn.conv2d_transpose(last_layer.output, self.weights, strides=strides, padding=padding, output_shape=out_shape)

            self.deconv3d = tf.layers.dropout(self.deconv3d)
            
            if normalizer is None:
                self.batch_norm = self.deconv3d
            else:
                self.batch_norm = normalizer(self.deconv3d, is_training=trainable)
                
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
                
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
        if not reuse:
            self.deconv3Vars.update({name: self.trainable_variables})
            
class Fully_Connected_down:
    def __init__(self,
                 deco,
                 i,
                 last_layer,
                 out_shape,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        self.deconv3Vars = {}
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            # if len(last_layer.output.shape.as_list())>2:
            #     last_layer.output=tf.reshape(last_layer.output,[last_layer.output.shape.as_list()[0],last_layer.output.shape.as_list()[1]*
            #                                       last_layer.output.shape.as_list()[2]*last_layer.output.shape.as_list()[3]])

            if len(out_shape)>2:
                out_size=out_shape[1] * out_shape[2] * out_shape[3]
                self.fc = fc(last_layer.output, out_size, activation_fn=self.activation)
            else:
                if deco == 1:
                    if i == 9:
                        self.fc = fc(last_layer, out_shape[-1], activation_fn=self.activation)
                    else:
                        self.fc = fc(last_layer.output, out_shape[-1], activation_fn=self.activation)
                else:
                    self.fc = fc(last_layer.output, out_shape[-1], activation_fn=self.activation)
            if normalizer is None:
                    self.batch_norm = self.fc
            else:
                    self.batch_norm = normalizer(self.fc, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
        if not reuse:
            self.deconv3Vars.update({name: self.trainable_variables})
class Fully_Connected_up:
    def __init__(self,
                 last_layer,
                 shape,
                 name,
                 activation=lrelu,
                 normalizer=tf.contrib.layers.batch_norm,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 trainable=True,
                 reuse=False
                 ):
        self.conv3Vars = {}
        temp = tf.trainable_variables()
        with tf.variable_scope(name, reuse=reuse):
            self.normalizer = normalizer
            self.activation = activation
            self.name = name
            if len(last_layer.output.shape.as_list())>2:
                last_layer.output=tf.reshape(last_layer.output,[last_layer.output.shape.as_list()[0],last_layer.output.shape.as_list()[1]*
                                                  last_layer.output.shape.as_list()[2]*last_layer.output.shape.as_list()[3]])
            self.fc = fc(last_layer.output, shape[0], activation_fn=self.activation)
            if normalizer is None:
                    self.batch_norm = self.fc
            else:
                    self.batch_norm = normalizer(self.fc, is_training=trainable)
            if activation is None:
                self.output = self.batch_norm
            else:
                self.output = activation(self.batch_norm)
        if trainable:
            self.trainable_variables = list(set(tf.trainable_variables()).symmetric_difference(set(temp)))
        else:
            self.trainable_variables = None
        if not reuse:
            self.conv3Vars.update({name: self.trainable_variables})


class UpAE:
    def __init__(self,
                 shape,
                 strides,
                 name,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 up_activation=lrelu,
                 up_normalizer=tf.contrib.layers.batch_norm,
                 up_padding="SAME",
                 up_trainable=True,
                 lr=0.0001,
                 beta=0.5
                 ):
        self.name = name
        self.shape = shape
        self.strides = strides
        self.up_activation = up_activation
        self.up_padding = up_padding
        self.initializer = initializer
        self.up_trainable = up_trainable
        self.up_normalizer = up_normalizer
        self.lr = lr
        self.beta = beta
        self.conv3 = {}
        self.conv3Vars = {}
        self.vars = {}
        self.optimizers = {}
        self.losses = {}
        self.input_layer = None

    def add_input_layer(self, input_layer):
        self.input_layer = input_layer


    def add_conv3(self, input_layer, reuse, name):
        if self.input_layer is None:
            self.add_input_layer(input_layer)
        temp = tf.trainable_variables()
        self.conv3.update({name: Conv3DLayer(last_layer=input_layer,
                                             shape=self.shape,
                                             strides=self.strides,
                                             name=self.name + '/conv3D',
                                             activation=self.up_activation,
                                             normalizer=self.up_normalizer,
                                             padding=self.up_padding,
                                             initializer=self.initializer,
                                             trainable=self.up_trainable,
                                             reuse=reuse)})
        if not reuse:
            self.conv3Vars.update({name: list(set(tf.trainable_variables()).symmetric_difference(set(temp)))})

    def add_fc_up(self,input_layer,reuse,name):
        if self.input_layer is None:
            self.add_input_layer(input_layer)
        temp = tf.trainable_variables()
        self.conv3.update({name: Fully_Connected_up(last_layer=input_layer,
                                             shape=self.shape,
                                             name=self.name + '/FC_up',
                                             activation=self.up_activation,
                                             normalizer=self.up_normalizer,
                                             initializer=self.initializer,
                                             trainable=self.up_trainable,
                                             reuse=reuse)})
        if not reuse:
            self.conv3Vars.update({name: list(set(tf.trainable_variables()).symmetric_difference(set(temp)))})


class DownAE:
    def __init__(self,
                 shape,
                 strides,
                 name,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 down_activation=lrelu,
                 down_normalizer=tf.contrib.layers.batch_norm,
                 down_padding="SAME",
                 down_trainable=True,
                 lr=0.0001,
                 lrc=0.0001,
                 beta=0.5
                 ):
        self.name = name
        self.shape = shape
        self.strides = strides
        self.down_activation = down_activation
        self.down_padding = down_padding
        self.initializer = initializer
        self.down_trainable = down_trainable
        self.down_normalizer = down_normalizer
        self.lr = lr
        self.lrc = lrc
        self.beta = beta
        self.deconv3 = {}
        self.deconv3Vars = {}
        self.vars = {}
        self.optimizers = {}
        self.losses = {}
        self.input_layer = None

    def add_input_layer(self, input_layer):
        self.input_layer = input_layer



    def add_fc_down(self,input_layer,reuse,name):
        assert self.input_layer is not None
        temp = tf.trainable_variables()
        self.deconv3.update({name: Fully_Connected_down(last_layer=input_layer,
                                             out_shape=self.input_layer.output.shape.as_list(),
                                             name=self.name + '/FC_down',
                                             activation=self.down_activation,
                                             normalizer=self.down_normalizer,
                                             initializer=self.initializer,
                                             trainable=self.down_trainable,
                                             reuse=reuse)})
        if not reuse:
            self.deconv3Vars.update({name: list(set(tf.trainable_variables()).symmetric_difference(set(temp)))})


    def add_deconv3(self, input_layer, reuse, name):
        # assert self.input_layer is not None
        temp = tf.trainable_variables()
        self.deconv3.update({name: Deconv3DLayer(last_layer=input_layer,
                                                 out_shape=input_layer.output.shape.as_list(),
                                                 shape=self.shape,
                                                 strides=self.strides,
                                                 name=self.name + '/deconv3D',
                                                 activation=self.down_activation,
                                                 normalizer=self.down_normalizer,
                                                 padding=self.down_padding,
                                                 initializer=self.initializer,
                                                 trainable=self.down_trainable,
                                                 reuse=reuse)})
        if not reuse:
            self.deconv3Vars.update({name: list(set(tf.trainable_variables()).symmetric_difference(set(temp)))})


def draw_point_cloud(input_points, canvasSize=256, space=100, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()
    output_points = points

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image, output_points


def point_cloud_one_views(points, angle=0.0):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    # +y is up direction
    # xrot is azimuth
    # yrot is in-plane
    # zrot is elevation

    img, output_point = draw_point_cloud(points, zrot=(90.0 + angle) / 180.0 * np.pi,
                                           xrot=45.0 / 180.0 * np.pi, yrot=(45.0) / 180.0 * np.pi)

    img = cv2.resize(img, (128,128))
    return img, output_point
