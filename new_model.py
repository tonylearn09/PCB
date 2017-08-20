import tensorflow as tf

class Model(object):
    def __init__(self, batch_size=128, learning_rate=1e-5, num_labels=1):
        self._batch_size = batch_size
        self._learning_rate = learning_rate
        self._num_labels = num_labels

    def inference(self, images, images_context, keep_prob, is_training):
        with tf.variable_scope('siamse') as scope:
            o1 = self.network(images, is_training=is_training)
            scope.reuse_variables()
            o2 = self.network(images_context, is_training=is_training)

        self.concat_res = tf.concat([o1, o2], 1)
        with tf.variable_scope('fully_connect') as scope:
            final = self.mlp(keep_prob, self.concat_res, is_training=is_training)
        return final


    def network(self, images, is_training):
        with tf.variable_scope('conv1') as scope:
            kernel = self._create_weights([3, 3, 3, 16])
            conv = self._create_conv2d(images, kernel, pad_mode='VALID')
            bias = self._create_bias([16])
            preactivation = tf.nn.bias_add(conv, bias)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            conv1 = tf.nn.relu(norm_preact, name=scope.name)
            self._activation_summary(conv1)

        # pool 1
        #h_pool1 = self._create_max_pool_2x2(conv1)

        with tf.variable_scope('conv2') as scope:
            kernel = self._create_weights([3, 3, 16, 16])
            conv = self._create_conv2d(conv1, kernel, pad_mode='SAME')
            #conv = self._create_conv2d(h_pool1, kernel)
            bias = self._create_bias([16])
            preactivation = tf.nn.bias_add(conv, bias)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            conv2 = tf.nn.relu(norm_preact, name=scope.name)
            self._activation_summary(conv2)

        # pool 2
        h_pool2 = self._create_max_pool_2x2(conv2)


        with tf.variable_scope('conv3') as scope:
            kernel = self._create_weights([3, 3, 16, 32])
            #conv = self._create_conv2d(conv2, kernel)
            conv = self._create_conv2d(h_pool2, kernel, pad_mode='SAME')
            bias = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv, bias)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            conv3 = tf.nn.relu(norm_preact, name=scope.name)
            self._activation_summary(conv3)

        with tf.variable_scope('conv4') as scope:
            kernel = self._create_weights([3, 3, 32, 32])
            conv = self._create_conv2d(conv3, kernel, pad_mode='VALID')
            bias = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv, bias)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            conv4 = tf.nn.relu(norm_preact, name=scope.name)
            self._activation_summary(conv4)


        with tf.variable_scope('conv5') as scope:
            kernel = self._create_weights([3, 3, 32, 32])
            conv = self._create_conv2d(conv4, kernel, pad_mode='VALID')
            bias = self._create_bias([32])
            preactivation = tf.nn.bias_add(conv, bias)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            conv5 = tf.nn.relu(norm_preact, name=scope.name)
            self._activation_summary(conv5)

        #h_pool5 = self._create_max_pool_2x2(conv5, pad_mode='VALID')
        #h_pool5 = tf.reshape(h_pool5, [tf.shape(h_pool5)[0], -1])
        # output shape: (None, 5, 5, 32)
        #print(h_pool5.get_shape())

        # shape (None, 11, 11, 32)
        conv5 = tf.reshape(conv5, [tf.shape(conv5)[0], -1])
        return conv5
        #return h_pool5

    def mlp(self, keep_prob, features, is_training):
        #print(features.get_shape())
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(features, [-1, 11 * 11 * 32 * 2])
            W_fc1 = self._create_weights([11 * 11 * 32 * 2, 512])
            b_fc1 = self._create_bias([512])
            preactivation = tf.nn.bias_add(tf.matmul(reshape, W_fc1), b_fc1)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            local1 = tf.nn.relu(norm_preact, name=scope.name)
            #local1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1, name=scope.name)
            self._activation_summary(local1)

        with tf.variable_scope('local2_linear') as scope:
            W_fc2 = self._create_weights([512, 256])
            b_fc2 = self._create_bias([256])
            local1_drop = tf.nn.dropout(local1, keep_prob)
            preactivation = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            local2 = tf.nn.relu(norm_preact, name=scope.name)
            #local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
            self._activation_summary(local2)


        with tf.variable_scope('local3_linear') as scope:
            W_fc3 = self._create_weights([256, 256])
            b_fc3 = self._create_bias([256])
            local2_drop = tf.nn.dropout(local2, keep_prob)
            preactivation = tf.nn.bias_add(tf.matmul(local2_drop, W_fc3), b_fc3)
            norm_preact = tf.contrib.layers.batch_norm(preactivation, center=True, 
                                                       scale=True, is_training=is_training)
            local3 = tf.nn.relu(norm_preact, name=scope.name)
            #local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
            self._activation_summary(local3)

        with tf.variable_scope('local4_linear') as scope:
            W_fc4 = self._create_weights([256, self._num_labels])
            b_fc4 = self._create_bias([self._num_labels])
            local3_drop = tf.nn.dropout(local3, keep_prob)
            local4 = tf.nn.bias_add(tf.matmul(local3_drop, W_fc4), b_fc4, name=scope.name)
            self._activation_summary(local4)


        return local4


    def train(self, loss, global_step):
        tf.summary.scalar('learning_rate', self._learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(self._learning_rate).minimize(loss, global_step=global_step)
        return train_op

    def loss(self, logits, labels):
        with tf.variable_scope('loss') as scope:
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
            #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=labels, pos_weight=0.9)
            cost = tf.reduce_mean(cross_entropy, name=scope.name)
            tf.summary.scalar('cost', cost)

        return cost

    def accuracy(self, logits, labels):
        with tf.variable_scope('accuracy') as scope:
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.nn.sigmoid(logits)), labels), dtype=tf.float32),
                                      name=scope.name)
            '''accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1)), dtype=tf.float32),
                                      name=scope.name)'''
            tf.summary.scalar('accuracy', accuracy)
        return accuracy

    def fnr_fpr(self, logits, labels):
        with tf.variable_scope('fp') as scope:
            predicted = tf.round(tf.nn.sigmoid(logits))
            actual = labels

            # Count true positives, true negatives, false positives and false negatives.
            tp = tf.count_nonzero(predicted * actual)
            tn = tf.count_nonzero((predicted - 1) * (actual - 1))
            fp = tf.count_nonzero(predicted * (actual - 1))
            fn = tf.count_nonzero((predicted - 1) * actual)

            # Calculate accuracy, precision, recall and F1 score.
            fnr = tf.to_float(fn) / tf.to_float(fn + tp)
            fpr = tf.to_float(fp) / tf.to_float(fp + tn)
            #precision = tp / (tp + fp)
            #recall = tp / (tp + fn)

            # Add metrics to TensorBoard.
            tf.summary.scalar('False_negative_rate', fnr)
            tf.summary.scalar('False_positive_rate', fpr)
        return fnr, fpr

    def _create_conv2d(self, x, W, pad_mode='SAME'):
        return tf.nn.conv2d(input=x,
                            filter=W,
                            strides=[1, 1, 1, 1],
                            padding=pad_mode)

    def _create_max_pool_2x2(self, input, pad_mode='SAME'):
        return tf.nn.max_pool(value=input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding=pad_mode)

    def _create_weights(self, shape):
        return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1, dtype=tf.float32))

    def _create_bias(self, shape):
        return tf.Variable(tf.constant(1., shape=shape, dtype=tf.float32))

    def _activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
