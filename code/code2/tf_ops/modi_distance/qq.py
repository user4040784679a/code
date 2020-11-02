import tensorflow as tf

tf.enable_eager_execution()


class Conv3DLayer:
    def __init__(self,
                 last_layer,
                 shape,
                 strides,
                 name,
                 activation=tf.nn.leaky_relu,
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


