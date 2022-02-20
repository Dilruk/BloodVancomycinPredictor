# TODO: 3 + scaled up the Y labels below to see if that improves resuls since labels were in 0-1 range
import copy
import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss
from time import time
import argparse
import LoadDataVanco_3 as DATA

#################### Arguments ####################

def parse_args():
    parser = argparse.ArgumentParser(description="Run Vanco Predictor.")
    parser.add_argument('--path', nargs='?', default='data_2/',
                        help='Input data path.')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size.')
    parser.add_argument('--fm_embedding_size', type=int, default=32,
                        help='Number of hidden factors in FM latent embeddings.')
    parser.add_argument('--userrep_size', type=int, default=32,
                        help='latent user representation size.')
    parser.add_argument('--nfm_layers', nargs='?', default='[64,32]',
                        help="Size of each layer in neuFM.")
    parser.add_argument('--fnn_layers', nargs='?', default='[64,32]',
                        help="Size of each layer in Forward Prediction NN.")
    parser.add_argument('--bnn_layers', nargs='?', default='[64,32]',
                        help="Size of each layer in Backward Prediction NN.")
    parser.add_argument('--nfm_dropout_layers', nargs='?', default='[0.3, 0.3]',
                        help="Dropout probability of each layer in NeuFM. 0 indicate no dropout.")
    parser.add_argument('--fnn_dropout_layers', nargs='?', default='[0.3, 0.3]',
                        help="Dropout probability of each layer in Forward prediction NN. 0 indicate no dropout.")
    parser.add_argument('--bnn_dropout_layers', nargs='?', default='[0.3, 0.3]',
                        help="Dropout probability of each layer in Backward prediction NN. 0 indicate no dropout.")
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdamOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--activation', nargs='?', default='relu',
                        help='Which activation function to use for deep layers: relu, sigmoid, tanh, identity')
    parser.add_argument('--early_stop', type=int, default=1,
                        help='Whether to perform early stop (0 or 1)')
    return parser.parse_args()


class VancoPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, epoch, batch_size, usr_x_size, fwd_x_size, bwd_x_size, fm_embedding_size, userrep_size, nfm_layers, fnn_layers, bnn_layers, loss_type, learning_rate, lamda_bilinear,
                 nfm_dropout_layers, fnn_dropout_layers, bnn_dropout_layers, optimizer_type, batch_norm, activation_function, verbose, early_stop, random_seed=2022):
        tf.compat.v1.disable_v2_behavior()
        # bind params to class
        self.epoch = epoch
        self.batch_size = batch_size
        self.usr_x_size = usr_x_size
        self.fwd_x_size = fwd_x_size
        self.bwd_x_size = bwd_x_size
        self.fm_embedding_size = fm_embedding_size
        self.userrep_size = userrep_size
        self.nfm_layers = nfm_layers
        self.fnn_layers = fnn_layers
        self.bnn_layers = bnn_layers
        self.loss_type = loss_type
        self.learning_rate = learning_rate
        self.lamda_bilinear = lamda_bilinear
        self.nfm_dropout_layers = np.array(nfm_dropout_layers)
        self.nfm_no_dropout = np.array([0 for i in range(len(nfm_dropout_layers))])
        self.fnn_dropout_layers = np.array(fnn_dropout_layers)
        self.fnn_no_dropout = np.array([0 for i in range(len(fnn_dropout_layers))])
        self.bnn_dropout_layers = np.array(bnn_dropout_layers)
        self.bnn_no_dropout = np.array([0 for i in range(len(bnn_dropout_layers))])
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.activation_function = activation_function
        self.verbose = verbose
        self.early_stop = early_stop
        self.random_seed = random_seed
        # performance of each epoch
        self.train_rmse_fwd, self.valid_rmse_fwd, self.test_rmse_fwd, self.train_rmse_bwd, self.valid_rmse_bwd, self.test_rmse_bwd = [], [], [], [], [], []

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.random.set_random_seed(self.random_seed)
            # Input data.
            self.userstate_features = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # None * usr_x_size
            self.forward_features = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # None * fwd_x_size
            self.forward_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.backward_features = tf.compat.v1.placeholder(tf.float32, shape=[None, None])  # None * bwd_x_size
            self.backward_labels = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.nfm_dropout_rate = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.fnn_dropout_rate = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.bnn_dropout_rate = tf.compat.v1.placeholder(tf.float32, shape=[None])
            self.train_phase = tf.compat.v1.placeholder(tf.bool)

            # Variables.
            self.weights, self.reg_list_nfm, self.reg_list_fnn, self.reg_list_bnn = self._initialize_weights()

            self.userstate = self.model1_NFM()  # None * userrep_size
            self.fwd_pred_y = self.model2_FWDPred()  # None * 1
            self.bwd_pred_y = self.model3_BWDPred()  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.fwd_loss = tf.nn.l2_loss(tf.subtract(self.forward_labels, self.fwd_pred_y)) + tf.keras.regularizers.l2(self.lamda_bilinear)(self.reg_list_fnn)  # regulizer
                    self.bwd_loss = tf.nn.l2_loss(tf.subtract(self.backward_labels, self.bwd_pred_y)) + tf.keras.regularizers.l2(self.lamda_bilinear)(self.reg_list_bnn)  # regulizer
                else:
                    self.fwd_loss = tf.nn.l2_loss(tf.subtract(self.forward_labels, self.fwd_pred_y))
                    self.bwd_loss = tf.nn.l2_loss(tf.subtract(self.backward_labels, self.bwd_pred_y))
            elif self.loss_type == 'log_loss':
                if self.lamda_bilinear > 0:
                    self.fwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.fwd_pred_y), self.forward_labels, weight=1.0, epsilon=1e-07, scope=None) + self.lamda_bilinear * (tf.nn.l2_loss(self.reg_list_fnn) + tf.nn.l2_loss(self.reg_list_nfm))  # regulizer
                    self.bwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.bwd_pred_y), self.backward_labels, weight=1.0, epsilon=1e-07, scope=None) + self.lamda_bilinear * (tf.nn.l2_loss(self.reg_list_bnn) + tf.nn.l2_loss(self.reg_list_nfm))  # regulizer
                else:
                    # self.fwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.fwd_pred_y), self.forward_labels, weight=1.0, epsilon=1e-07, scope=None)
                    # self.bwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.bwd_pred_y), self.backward_labels, weight=1.0, epsilon=1e-07, scope=None)

                    self.fwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.fwd_pred_y), self.forward_labels)
                    self.bwd_loss = tf.compat.v1.losses.log_loss(tf.sigmoid(self.bwd_pred_y), self.backward_labels)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.fwd_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.fwd_loss)
                self.bwd_optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.bwd_loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.fwd_optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.fwd_loss)
                self.bwd_optimizer = tf.compat.v1.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.bwd_loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.fwd_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.fwd_loss)
                self.bwd_optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.bwd_loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.fwd_optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.fwd_loss)
                self.bwd_optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.bwd_loss)

            # init
            self.saver = tf.compat.v1.train.Saver()
            init = tf.compat.v1.global_variables_initializer()
            self.sess = tf.compat.v1.Session()
            self.sess.run(init)

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim  # .value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    # Model 1 : Learns latent representations (St) for each user. Output is used in both forward and backward prediction models.
    def model1_NFM(self):
        self.expanded_userstate_features = tf.keras.backend.repeat_elements(tf.expand_dims(self.userstate_features, 2), self.fm_embedding_size, axis=2)
        self.weighted_embeddings = tf.multiply(self.weights['nfm_embeddings'], self.expanded_userstate_features)    # [None x user_x_size x embedding_size]

        # ________ FM __________
        # self.FM_o1 = tf.tensordot(self.userstate_features, self.weights['nfm_feature_bias'], axes=[1, 0])  # None * 1
        self.FM_o1 = tf.reduce_sum(self.weighted_embeddings, axis=1)  # None * embedding_size

        # _________ sum_square _____________
        # get the summed up embeddings of features.
        self.summed_features_emb = tf.reduce_sum(self.weighted_embeddings, 1)  # None * K
        self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

        # _________ square_sum _____________
        self.squared_features_emb = tf.square(self.weighted_embeddings)
        self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

        self.FM_o2 = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
        # if self.batch_norm:
        #     self.FM_o2 = self.batch_norm_layer(self.FM_o2, train_phase=self.train_phase, scope_bn='bn_nfm')
        # self.FM_o2 = tf.nn.dropout(self.FM_o2, self.nfm_dropout_rate[-1])  # dropout at the bilinear interactin layer

        # Added o1 and o2 together
        self.FM_o2 = tf.reduce_sum([self.FM_o1, self.FM_o2], axis=0)            # None * K
        # ________ Deep NFM Layers __________
        for i in range(0, len(self.nfm_layers)):
            self.FM_o2 = tf.add(tf.matmul(self.FM_o2, self.weights['nfm_layer_%d' % i]), self.weights['nfm_bias_%d' % i])  # None * nfm_layer[i] * 1
            if self.batch_norm:
                self.FM_o2 = self.batch_norm_layer(self.FM_o2, train_phase=self.train_phase, scope_bn='bn_nfm_%d' % i)  # None * nfm_layer[i] * 1
            self.FM_o2 = self.activation_function(self.FM_o2)
            # self.FM_o2 = tf.nn.relu(self.FM_o2)
            self.FM_o2 = tf.nn.dropout(self.FM_o2, rate=self.nfm_dropout_rate[i])  # dropout at each Deep layer

        # _________out _________
        u_state = tf.add(tf.matmul(self.FM_o2, self.weights['nfm_user_rep_layer']), self.weights['nfm_user_rep_bias'])  # None * userrep_size

        return u_state

    # Model 2 : Forward Prediction.
    def model2_FWDPred(self):

        # Model.
        self.model2_h = tf.concat([self.userstate, self.forward_features], axis=1)  # None * (32 + 16)

        # ________ Deep FWD Layers __________
        for i in range(0, len(self.fnn_layers)):
            self.model2_h = tf.add(tf.matmul(self.model2_h, self.weights['fnn_layer_%d' % i]), self.weights['fnn_bias_%d' % i])  # None * fnn_layer[i] * 1
            if self.batch_norm:
                self.model2_h = self.batch_norm_layer(self.model2_h, train_phase=self.train_phase, scope_bn='bn_fnn_%d' % i)  # None * fnn_layer[i] * 1
            self.model2_h = self.activation_function(self.model2_h)
            # self.model2_h = tf.nn.relu(self.model2_h)
            self.model2_h = tf.nn.dropout(self.model2_h, rate=self.fnn_dropout_rate[i])  # dropout at each Deep layer

        # _________out _________
        fwd_pred_y = tf.add(tf.matmul(self.model2_h, self.weights['fnn_prediction']), self.weights['fnn_prediction_bias'])  # None * 1
        return fwd_pred_y

    # Model 2 : Backward Prediction.
    def model3_BWDPred(self):

        # Model.
        self.model3_h = tf.concat([self.userstate, self.backward_features], axis=1)

        # ________ Deep BWD Layers __________
        for i in range(0, len(self.bnn_layers)):
            self.model3_h = tf.add(tf.matmul(self.model3_h, self.weights['bnn_layer_%d' % i]), self.weights['bnn_bias_%d' % i])  # None * bnn_layer[i] * 1
            if self.batch_norm:
                self.model3_h = self.batch_norm_layer(self.model3_h, train_phase=self.train_phase, scope_bn='bn_bnn_%d' % i)  # None * fnn_layer[i] * 1
            self.model3_h = self.activation_function(self.model3_h)
            self.model3_h = tf.nn.dropout(self.model3_h, rate=self.bnn_dropout_rate[i])  # dropout at each Deep layer

        # _________out _________
        bwd_pred_y = tf.add(tf.matmul(self.model3_h, self.weights['bnn_prediction']), self.weights['bnn_prediction_bias'])   # None * 1


        return bwd_pred_y

    def _initialize_weights(self):

        all_weights = dict()

        # Lists of terms for regularization
        reg_list_nfm, reg_list_fnn, reg_list_bnn = [], [], []
        all_weights['nfm_embeddings'] = tf.Variable(tf.random.normal([self.usr_x_size, self.fm_embedding_size], 0.0, 0.01), name='fm_embeddings')  # usr_x_size * K
        all_weights['nfm_feature_bias'] = tf.Variable(tf.random.uniform([self.usr_x_size, 1], 0.0, 0.0), name='fm_bias')  # usr_x_size * 1
        reg_list_nfm.append(all_weights['nfm_embeddings'])

        # deep nfm layers
        num_nfm_layer = len(self.nfm_layers)
        if num_nfm_layer > 0:
            # glorot = np.sqrt(2.0 / (self.fm_embedding_size + self.nfm_layers[0]))
            all_weights['nfm_layer_0'] = tf.Variable(tf.random.normal([self.fm_embedding_size, self.nfm_layers[0]], 0.0, 0.01), name='nfm_layer_0', dtype=np.float32)
            all_weights['nfm_bias_0'] = tf.Variable(tf.random.normal([self.nfm_layers[0]], 0.0, 0.01), name='nfm_bias_0', dtype=np.float32)  # 1 * layers[0]
            reg_list_nfm.append(all_weights['nfm_layer_0'])
            for i in range(1, num_nfm_layer):
                # glorot = np.sqrt(2.0 / (self.nfm_layers[i - 1] + self.nfm_layers[i]))
                all_weights['nfm_layer_%d' % i] = tf.Variable(tf.random.normal([self.nfm_layers[i - 1], self.nfm_layers[i]], 0.0, 0.01), name='nfm_layer_%d' % i, dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['nfm_bias_%d' % i] = tf.Variable(tf.random.normal([self.nfm_layers[i]], 0.0, 0.01), name='nfm_bias_%d' % i, dtype=np.float32)  # 1 * layer[i]
                reg_list_nfm.append(all_weights['nfm_layer_%d' % i])
            # prediction layer
            # glorot = np.sqrt(2.0 / (self.nfm_layers[-1] + 1))
            all_weights['nfm_user_rep_layer'] = tf.Variable(tf.random.normal([self.nfm_layers[-1], self.userrep_size], 0.0, 0.01), name='nfm_user_rep_layer', dtype=np.float32)  # layers[-1] * userrep_size
        else:
            all_weights['nfm_user_rep_layer'] = tf.Variable(tf.random.normal([self.fm_embedding_size, self.userrep_size], 0.0, 0.01), name='nfm_user_rep_layer', dtype=np.float32)  # hidden_factor * 1  # hidden_factor * userrep_size
        all_weights['nfm_user_rep_bias'] = tf.Variable([self.userrep_size], name='nfm_user_rep_bias', dtype=np.float32)  # 1
        reg_list_nfm.append(all_weights['nfm_user_rep_layer'])

        # deep forward nn layers
        num_fnn_layer = len(self.fnn_layers)
        if num_fnn_layer > 0:
            # glorot = np.sqrt(2.0 / (self.userrep_size + self.fnn_layers[0]))
            all_weights['fnn_layer_0'] = tf.Variable(tf.random.normal([self.userrep_size + self.fwd_x_size, self.fnn_layers[0]], 0.0, 0.01), name='fnn_layer_0', dtype=np.float32)
            all_weights['fnn_bias_0'] = tf.Variable(tf.random.normal([self.fnn_layers[0]], 0.0, 0.01), name='fnn_bias_0', dtype=np.float32)  # 1 * layers[0]
            reg_list_fnn.append(all_weights['fnn_layer_0'])
            for i in range(1, num_fnn_layer):
                # glorot = np.sqrt(2.0 / (self.fnn_layers[i - 1] + self.fnn_layers[i]))
                all_weights['fnn_layer_%d' % i] = tf.Variable(tf.random.normal([self.fnn_layers[i - 1], self.fnn_layers[i]], 0.0, 0.01), name='fnn_layer_%d' % i, dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['fnn_bias_%d' % i] = tf.Variable(tf.random.normal([self.fnn_layers[i]], 0.0, 0.01), name='fnn_bias_%d' % i, dtype=np.float32)  # 1 * layer[i]
                reg_list_fnn.append(all_weights['fnn_layer_%d' % i])
            # prediction layer
            # glorot = np.sqrt(2.0 / (self.fnn_layers[-1] + 1))
            all_weights['fnn_prediction'] = tf.Variable(tf.random.normal([self.fnn_layers[-1], 1], 0.0, 0.01), name='fnn_prediction', dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['fnn_prediction'] = tf.Variable(tf.random.normal([self.userrep_size + self.fwd_x_size, 1], 0.0, 0.01), name='fnn_prediction', dtype=np.float32)  # hidden_factor * 1

        all_weights['fnn_prediction_bias'] = tf.Variable([1], name='fnn_prediction_bias', dtype=np.float32)  # 1
        reg_list_fnn.append(all_weights['fnn_prediction'])

        # deep backward nn layers
        num_bnn_layer = len(self.bnn_layers)
        if num_bnn_layer > 0:
            # glorot = np.sqrt(2.0 / (self.userrep_size + self.bnn_layers[0]))
            all_weights['bnn_layer_0'] = tf.Variable(tf.random.normal([self.userrep_size + self.bwd_x_size, self.bnn_layers[0]], 0.0, 0.01), name='bnn_layer_0', dtype=np.float32)
            all_weights['bnn_bias_0'] = tf.Variable(tf.random.normal([self.bnn_layers[0]], 0.0, 0.01), name='bnn_bias_0', dtype=np.float32)  # 1 * layers[0]
            reg_list_bnn.append(all_weights['bnn_layer_0'])
            for i in range(1, num_bnn_layer):
                # glorot = np.sqrt(2.0 / (self.bnn_layers[i - 1] + self.bnn_layers[i]))
                all_weights['bnn_layer_%d' % i] = tf.Variable(tf.random.normal([self.bnn_layers[i - 1], self.bnn_layers[i]], 0.0, 0.01), name='bnn_layer_%d' % i, dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bnn_bias_%d' % i] = tf.Variable(tf.random.normal([self.bnn_layers[i]], 0.0, 0.01), name='bnn_bias_%d' % i, dtype=np.float32)  # 1 * layer[i]
                reg_list_bnn.append(all_weights['bnn_layer_%d' % i])
            # prediction layer
            # glorot = np.sqrt(2.0 / (self.bnn_layers[-1] + 1))
            all_weights['bnn_prediction'] = tf.Variable(tf.random.normal([self.bnn_layers[-1], 1], 0.0, 0.01), name='bnn_prediction', dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['bnn_prediction'] = tf.Variable(tf.random.normal([self.userrep_size + self.bwd_x_size, 1], name='bnn_prediction', dtype=np.float32))  # hidden_factor * 1

        all_weights['bnn_prediction_bias'] = tf.Variable([1], name='bnn_prediction_bias', dtype=np.float32)  # 1 * layers[-1]
        reg_list_bnn.append(all_weights['bnn_prediction'])
        return all_weights, reg_list_nfm, reg_list_fnn, reg_list_bnn

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # bn_train = tf.keras.layers.BatchNormalization(x, momentum=0.9, center=True, scale=True, trainable=True)
        # bn_inference = tf.keras.layers.BatchNormalization(x, momentum=0.9, center=True, scale=True, trainable=True)
        bn_train = tf.compat.v1.layers.batch_normalization(x, momentum=0.9, center=True, scale=True, trainable=True)
        bn_inference = tf.compat.v1.layers.batch_normalization(x, momentum=0.9, center=True, scale=True, trainable=True)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)

        # tf.keras.layers.BatchNormalization(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=True, reuse=None, trainable=True, scope=scope_bn)

        # tf.compat.v1.layers.batch_normalization
        return z

    def partial_fit_fwd(self, data):  # fit a batch to forward prediction nn
        feed_dict = {self.userstate_features: data['FWD_User_X'], self.forward_features: data['FWD_X'], self.forward_labels: data['FWD_Y'], self.nfm_dropout_rate: self.nfm_dropout_layers, self.fnn_dropout_rate: self.fnn_dropout_layers, self.train_phase: True}
        fwd_loss, fwd_opt = self.sess.run((self.fwd_loss, self.fwd_optimizer), feed_dict=feed_dict)
        return fwd_loss

    def partial_fit_bwd(self, data):  # fit a batch to backward prediction nn
        feed_dict = {self.userstate_features: data['BWD_User_X'], self.backward_features: data['BWD_X'], self.backward_labels: data['BWD_Y'], self.nfm_dropout_rate: self.nfm_dropout_layers, self.bnn_dropout_rate: self.bnn_dropout_layers, self.train_phase: True}
        bwd_loss, bwd_opt = self.sess.run((self.bwd_loss, self.bwd_optimizer), feed_dict=feed_dict)
        return bwd_loss

    def get_random_block_from_data(self, prefix, data_user_X, data_X, data_Y, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data_Y) - batch_size)
        user_X, X, Y = [], [], []
        # get sample from front
        i = start_index
        while len(X) < batch_size and i < len(data_X):
            if len(data_X[i]) == len(data_X[start_index]):
                user_X.append(data_user_X[i])
                X.append(data_X[i])
                Y.append(data_Y[i])
                i = i + 1
            else:
                break
        # get sample from back
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data_X[i]) == len(data_X[start_index]):
                user_X.append(data_user_X[i])
                X.append(data_X[i])
                Y.append(data_Y[i])
                i = i - 1
            else:
                break
        return {prefix + 'User_X': user_X, prefix + 'X': X, prefix + 'Y': Y}

    def shuffle_in_unison_scary(self, a, b, c, d, e, f):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)
        np.random.set_state(rng_state)
        np.random.shuffle(d)
        np.random.set_state(rng_state)
        np.random.shuffle(e)
        np.random.set_state(rng_state)
        np.random.shuffle(f)

    def train(self, Train_data, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        # if self.verbose > 0:
            # t2 = time()
            # init_train_fwd = self.evaluate_fwd(Train_data)
            # init_valid_fwd = self.evaluate_fwd(Validation_data)
            # init_test_fwd = self.evaluate_fwd(Test_data)
            #
            # init_train_bwd = self.evaluate_bwd(Train_data)
            # init_valid_bwd = self.evaluate_bwd(Validation_data)
            # init_test_bwd = self.evaluate_bwd(Test_data)
            #
            # print("Init FWD: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" % (init_train_fwd, init_valid_fwd, init_test_fwd, time() - t2))
            # print("Init BWD: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" % (init_train_bwd, init_valid_bwd, init_test_bwd, time() - t2))

        print("Epoch\ttrain_fwd\tvalidation_fwd\ttest_fwd\ttrain_bwd\tvalidation_bwd\ttest_bwd")

        for epoch in range(self.epoch):
            t1 = time()
            # self.shuffle_in_unison_scary(Train_data['FWD_User_X'], Train_data['FWD_X'], Train_data['FWD_Y'], Train_data['BWD_User_X'], Train_data['BWD_X'], Train_data['BWD_Y'])

            total_batch_FWD = int(len(Train_data['FWD_Y']) / self.batch_size)
            total_batch_BWD = int(len(Train_data['BWD_Y']) / self.batch_size)

            for i in range(total_batch_FWD):
                # generate a batch
                batch_xs_fwd = self.get_random_block_from_data('FWD_', Train_data['FWD_User_X'], Train_data['FWD_X'], Train_data['FWD_Y'], self.batch_size)
                # Fit training
                self.partial_fit_fwd(batch_xs_fwd)
            for i in range(total_batch_BWD):
                # generate a batch
                batch_xs_bwd = self.get_random_block_from_data('BWD_', Train_data['BWD_User_X'], Train_data['BWD_X'], Train_data['BWD_Y'], self.batch_size)
                # Fit training
                self.partial_fit_bwd(batch_xs_bwd)
            # for i in range(total_batch_FWD):
            #     # generate a batch
            #     batch_xs_fwd = self.get_random_block_from_data('FWD_', Train_data['FWD_User_X'], Train_data['FWD_X'], Train_data['FWD_Y'], self.batch_size)
            #     # Fit training
            #     self.partial_fit_fwd(batch_xs_fwd)
            t2 = time()

            # output validation
            train_result_fwd = self.evaluate_fwd(Train_data)
            valid_result_fwd = self.evaluate_fwd(Validation_data)
            test_result_fwd = self.evaluate_fwd(Test_data)
            train_result_bwd = self.evaluate_bwd(Train_data)
            valid_result_bwd = self.evaluate_bwd(Validation_data)
            test_result_bwd = self.evaluate_bwd(Test_data)

            self.train_rmse_fwd.append(train_result_fwd)
            self.valid_rmse_fwd.append(valid_result_fwd)
            self.test_rmse_fwd.append(test_result_fwd)
            self.train_rmse_bwd.append(train_result_bwd)
            self.valid_rmse_bwd.append(valid_result_bwd)
            self.test_rmse_bwd.append(test_result_bwd)

            if self.verbose > 0 and epoch % self.verbose == 0:
                print("%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f"
                      % (epoch + 1, train_result_fwd, valid_result_fwd, test_result_fwd, train_result_bwd, valid_result_bwd, test_result_bwd))
            if self.early_stop > 0 and self.eva_termination(self.valid_rmse_fwd) and self.eva_termination(self.valid_rmse_bwd):
                # print "Early stop at %d based on validation result." %(epoch+1)
                break

        # if epoch == self.epoch-1:
        self.saver.save(self.sess, './savedmodels/Vanco-Predictor-model' + str(time()), global_step=0)
        # self.saver.save(self.userstate, '.Vanco-Predictor-model' + str(time()), global_step=0)

    def getstats(self, data):
        feed_dict = {self.userstate_features: data['FWD_User_X'], self.forward_features: data['FWD_X'], self.forward_labels: data['FWD_Y'], self.nfm_dropout_rate: self.nfm_no_dropout, self.fnn_dropout_rate: self.fnn_no_dropout, self.train_phase: False}
        user_states = self.sess.run(self.userstate, feed_dict=feed_dict)

        return user_states, copy.deepcopy(data['FWD_subject_ids']), copy.deepcopy(data['FWD_hadm_ids'])

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate_fwd(self, data):  # evaluate the forward prediction results for an input set
        num_example = len(data['FWD_Y'])
        feed_dict = {self.userstate_features: data['FWD_User_X'], self.forward_features: data['FWD_X'], self.forward_labels: data['FWD_Y'], self.nfm_dropout_rate: self.nfm_no_dropout, self.fnn_dropout_rate: self.fnn_no_dropout, self.train_phase: False}
        predictions = self.sess.run(self.fwd_pred_y, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['FWD_Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)  # I haven't checked the log_loss
            return logloss

    def evaluate_bwd(self, data):  # evaluate the backward prediction results for an input set
        num_example = len(data['BWD_Y'])
        feed_dict = {self.userstate_features: data['BWD_User_X'], self.backward_features: data['BWD_X'], self.backward_labels: data['BWD_Y'], self.nfm_dropout_rate: self.nfm_no_dropout, self.bnn_dropout_rate: self.bnn_no_dropout, self.train_phase: False}
        predictions = self.sess.run(self.bwd_pred_y, feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['BWD_Y'], (num_example,))
        if self.loss_type == 'square_loss':
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)  # TODO: check the log_loss
            return logloss


if __name__ == '__main__':
    # Data loading
    args = parse_args()
    activation_function = tf.nn.relu
    if args.activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif args.activation == 'tanh':
        activation_function = tf.tanh
    elif args.activation == 'identity':
        activation_function = tf.identity

    if args.verbose > 0:
        print("Neural FM: fm_embedding_size=%d, userrep_size=%d, nfm_layers=%s, fnn_layers=%s, bnn_layers=%s, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.4f, nfm_dropoutlayers=%s, fnn_dropout_layers=%s, bnn_dropout_layers=%s, optimizer=%s, batch_norm=%d, activation=%s, early_stop=%d"
              % (args.fm_embedding_size, args.userrep_size, args.nfm_layers, args.fnn_layers, args.bnn_layers, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda, args.nfm_dropout_layers, args.fnn_dropout_layers, args.bnn_dropout_layers, args.optimizer, args.batch_norm, args.activation, args.early_stop))

    data = DATA.LoadData(path=args.path)

    # TODO: Scaling label values
    FWD_Y_SCALE_VALUE = 100
    BWD_Y_SCALE_VALUE = 10

    data.Train_data['FWD_Y'] = [[FWD_Y_SCALE_VALUE * y] for [y] in data.Train_data['FWD_Y']]
    data.Train_data['BWD_Y'] = [[BWD_Y_SCALE_VALUE * y] for [y] in data.Train_data['BWD_Y']]

    data.Validation_data['FWD_Y'] = [[FWD_Y_SCALE_VALUE * y] for [y] in data.Validation_data['FWD_Y']]
    data.Validation_data['BWD_Y'] = [[BWD_Y_SCALE_VALUE * y] for [y] in data.Validation_data['BWD_Y']]

    data.Test_data['FWD_Y'] = [[FWD_Y_SCALE_VALUE * y] for [y] in data.Test_data['FWD_Y']]
    data.Test_data['BWD_Y'] = [[BWD_Y_SCALE_VALUE * y] for [y] in data.Test_data['BWD_Y']]

    # Training
    t1 = time()
    model = VancoPredictor(args.epoch, args.batch_size, data.no_userstate_features, data.no_forward_features, data.no_backward_features, args.fm_embedding_size, args.userrep_size, eval(args.nfm_layers), eval(args.fnn_layers), eval(args.bnn_layers), args.loss_type, args.lr, args.lamda, eval(args.nfm_dropout_layers), eval(args.fnn_dropout_layers), eval(args.bnn_dropout_layers), args.optimizer, args.batch_norm, activation_function, args.verbose, args.early_stop)
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    # Find the best validation result across iterations
    best_valid_score_fwd = 0
    best_valid_score_bwd = 0
    if args.loss_type == 'square_loss':
        best_valid_score_fwd = min(model.valid_rmse_fwd)
        best_valid_score_bwd = min(model.valid_rmse_bwd)
    elif args.loss_type == 'log_loss':
        best_valid_score_fwd = max(model.valid_rmse_fwd)
        best_valid_score_bwd = max(model.valid_rmse_bwd)
    best_epoch_fwd = model.valid_rmse_fwd.index(best_valid_score_fwd)
    best_epoch_bwd = model.valid_rmse_bwd.index(best_valid_score_bwd)
    print("FWD Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
          % (best_epoch_fwd + 1, model.train_rmse_fwd[best_epoch_fwd], model.valid_rmse_fwd[best_epoch_fwd], model.test_rmse_fwd[best_epoch_fwd], time() - t1))
    print("BWD Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]"
          % (best_epoch_bwd + 1, model.train_rmse_bwd[best_epoch_bwd], model.valid_rmse_bwd[best_epoch_bwd], model.test_rmse_bwd[best_epoch_bwd], time() - t1))

    train_fwd_user_states, train_fwd_subject_ids, train_fwd_hadm_ids = model.getstats(data.Train_data)
    np.save('userstates.npy', list(train_fwd_user_states))
    np.save('subject_ids.npy', list(data.train_fwd_subject_ids))
    np.save('hadm_ids.npy', list(data.train_fwd_hadm_ids))

    tf.keras.backend.clear_session()

# TODO: Increase BWD features? Only 4
