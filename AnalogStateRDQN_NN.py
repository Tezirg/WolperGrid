# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka

from grid2op.Converter import AnalogStateConverter as analog

class AnalogStateRDQN_NN(object):
    def __init__(self,
                 q_size,
                 topology_size,
                 line_size,
                 dispatch_size,
                 observation_size,
                 n_bus = 2,                 
                 learning_rate = 1e-5,
                 is_training = False):
        self.q_size = q_size
        self.n_bus = n_bus
        self.topo_size = topology_size
        self.n_line = line_size
        self.disp_size = dispatch_size
        self.observation_size = observation_size
        self.encoded_size = 384
        self.h_size = 512
        self.lr = learning_rate
        self.is_training = is_training
        self.model = None
        self.construct_q_network()

    def forward_encode(self, input_layer, name):
        batch_size = tf.shape(input_layer)[0]
        trace_size = tf.shape(input_layer)[1]
        obs_size = self.observation_size
        # Reshape to (batch_size * trace_len, data_size)
        # So that Dense process each data vect separately
        input_shape = (-1, obs_size)
        input2d = tf.reshape(input_layer, input_shape,
                             name=name+"_input_reshape")

        # Bayesian NN simulate using dropout
        lay1 = tfkl.Dropout(self.dropout_rate, name=name+"_bnn")(input2d)

        # Five layers encoder
        lay1 = tfkl.Dense(self.encoded_size + 128, name=name+"_fc1")(lay1)
        lay1 = tf.nn.leaky_relu(lay1, alpha=0.01, name=name+"_leak_fc1")

        lay2 = tfkl.Dense(self.encoded_size + 64, name=name+"_fc2")(lay1)
        lay2 = tf.nn.leaky_relu(lay2, alpha=0.01, name=name+"_leak_fc2")

        lay3 = tfkl.Dense(self.encoded_size + 32, name=name+"_fc3")(lay2)
        lay3 = tf.nn.leaky_relu(lay3, alpha=0.01, name=name+"_leak_fc3")
        
        lay4 = tfkl.Dense(self.encoded_size + 16, name=name+"_fc4")(lay3)
        lay4 = tf.nn.leaky_relu(lay4, alpha=0.01, name=name+"_leak_fc4")

        lay5 = tfkl.Dense(self.encoded_size, name=name+"_fc5")(lay4)
        lay5 = tf.nn.leaky_relu(lay5, alpha=0.01, name=name+"_leak_fc5")

        # Reshape encoded to (batch_size, trace_len, encoded_size)
        encoded_shape = (batch_size, trace_size, self.encoded_size)
        encoded = tf.reshape(lay5, encoded_shape, name=name+"_encoded")
        return encoded

    def forward_rnn(self, input_rnn, mem_state, carry_state, name):
        # Single LSTM cell over trace_length
        lstm_layer = tfkl.LSTM(self.h_size, return_state=True,
                               name=name+"_lstm")
        states = [mem_state, carry_state]
        lstm_out, mem, carry = lstm_layer(input_rnn, initial_state=states)
        return lstm_out, mem, carry

    def forward_vec(self, hidden, out_size, name):
        # Always add some noise to spread outputs
        noisy = tfkl.GaussianDropout(0.2)(hidden, self.is_training)
        
        # Decode to partial state vec
        vec_1 = tfkl.Dense(out_size * 2, name=name+"_fc1_vec")(noisy)
        vec_1 = tf.nn.leaky_relu(vec_1, alpha=0.01,
                                 name=name+"_leak1_vec")
        vec_2 = tfkl.Dense(out_size + out_size // 2,
                           name=name+"_fc2_vec")(vec_1)
        vec_2 = tf.nn.leaky_relu(vec_2, alpha=0.01,
                                 name=name+"_leak2_vec")
        vec = tfkl.Dense(out_size, name=name+"_fc3_vec")(vec_2)
        return vec
        
    def forward_streams(self, hidden, out_size, name):
        # Advantage stream
        advantage = tfkl.Dense(128, name=name+"_fcadv")(hidden)
        advantage = tf.nn.leaky_relu(advantage, alpha=0.01,
                                     name=name+"_leak_adv")
        advantage = tfkl.Dense(out_size, name=name+"_adv")(advantage)
        advantage_mean = tf.math.reduce_mean(advantage, axis=1,
                                             keepdims=True,
                                             name=name+"_adv_mean")
        advantage = tfkl.subtract([advantage, advantage_mean],
                                  name= name+"_adv_sub")

        # Value stream
        value = tfkl.Dense(128, name=name+"_fcval")(hidden)
        value = tf.nn.leaky_relu(value, alpha=0.01, name=name+"_leak_val")
        value = tfkl.Dense(1, name=name+"_val")(value)

        # Q values = val + adv
        q = tf.math.add(value, advantage, name=name+"_out")
        return q

    def construct_q_network(self):
        # Defines input tensors and scalars
        self.trace_length = tf.Variable(1, trainable=False,
                                        dtype=tf.int32, name="trace_len")
        self.dropout_rate = tf.Variable(0.0, trainable=False,
                                        dtype=tf.float32, name="drop_rate")
        states_shape = (self.h_size,)
        input_mem = tfk.Input(dtype=tf.float32, shape=states_shape,
                              name='input_mem_state')
        input_carry = tfk.Input(dtype=tf.float32, shape=states_shape,
                                name='input_carry_state')
        input_shape = (None, self.observation_size)
        input_layer = tfk.Input(dtype=tf.float32, shape=input_shape,
                                name='input_obs')

        # Forward encode
        encoded = self.forward_encode(input_layer, "obs")
        
        # Forward recurring part
        hidden, mem, carry = self.forward_rnn(encoded,
                                              input_mem,
                                              input_carry,
                                              "rnn")        
        # Topology buses
        t_size = self.q_size * self.topo_size * self.n_bus
        t_shape = (-1, self.q_size, self.n_bus, self.topo_size)
        t_vec = self.forward_vec(hidden, t_size, "bus")
        print ("t vec shape=", t_vec.shape)
        t = tf.reshape(t_vec, t_shape, name="bus_shape")
        t = tf.nn.softmax(t, axis=-2, name="bus_softmax")
        print ("t shape=", t.shape)

        # Lines
        l_size = self.q_size * self.n_line
        l_shape = (-1, self.q_size, self.n_line)
        l_vec = self.forward_vec(hidden, l_size, "line")
        l_vec = tf.nn.relu(l_vec, name="line_relu")
        l = tf.reshape(l_vec, l_shape, name="shape_line")
        print("l shape =", l.shape)

        # Redispatch
        d_size = self.q_size * self.disp_size
        d_shape = (-1, self.q_size, self.disp_size)
        d_vec = self.forward_vec(hidden, d_size, "disp")
        d_vec = tf.nn.tanh(d_vec, name="tanh_disp")
        print ("d_vec shape=", d_vec.shape)
        d = tf.reshape(d_vec, d_shape, name="shape_disp")
        print ("d shape=", d.shape)

        # Advantage and Value streams
        qt = tfkl.Flatten(name="q_b_flat")(t)
        ql = tfkl.Flatten(name="q_l_flat")(l)
        qd = tfkl.Flatten(name="q_d_flat")(d)
        h_concat = tf.concat([qt, ql, qd, hidden], 1, name="Q_concat")
        print ("concat", h_concat.shape)
        q = self.forward_streams(h_concat, self.q_size, "Q")
        print ("q shape =", q.shape)

        # Backwards pass
        model_inputs = [
            input_mem,
            input_carry,
            input_layer
        ]
        model_outputs = [
            q,
            t, l, d,
            mem, carry
        ]
        self.model = tfk.Model(inputs=model_inputs,
                               outputs=model_outputs,
                               name=self.__class__.__name__)
        losses = [
            self._clipped_mse_loss,
            self._clipped_mse_loss, # Should be bce or cosine sim
            self._clipped_mse_loss, # mse OK i guess
            self._clipped_mse_loss, # Should be sh
            self._no_loss,
            self._no_loss
        ]
        self.optimizer = tfko.Adam(lr=self.lr, clipnorm=1.0)
        self.model.compile(loss=losses, optimizer=self.optimizer)

    def _no_loss(self, y_true, y_pred):
        return 0.0

    def _clipped_bce_loss(self, y_true, y_pred):
        loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred),
                                   name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1e2, name="loss_clip")
        return clipped_loss

    def _clipped_mse_loss(self, y_true, y_pred):
        loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred),
                                   name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1e2, name="loss_clip")
        return clipped_loss

    def bayesian_move(self, data, mem, carry, rate = 0.0):
        self.dropout_rate.assign(float(rate))
        self.trace_length.assign(1)

        input_shape = (1, 1, self.observation_size)
        data_input = data.reshape(input_shape)
        mem_input = mem.reshape(1, self.h_size)
        carry_input = carry.reshape(1, self.h_size)
        model_input = [mem_input, carry_input, data_input]

        pred = self.model.predict(model_input, batch_size = 1)
        # Model has 6 outputs, and batch size 1
        q = pred[0][0] # n_grid x q
        bus = pred[1][0] # n_grid x n_bus x dim_topo
        line = pred[2][0] # n_grid x n_line 
        disp = pred[3][0] # n_grid x n_gen

        # Get index of highest q value
        grid = np.argmax(q)

        # Get action from index
        q = q[grid]
        bus = bus[grid]
        line = line[grid]
        disp = disp[grid]
        return (grid, bus, line, disp, q), pred[4], pred[5]

    def predict_move(self, data, mem, carry):
        return self.bayesian_move(data, mem, carry, 0.0)

    def random_move(self, data, mem, carry, obs):
        self.trace_length.assign(1)
        self.dropout_rate.assign(0.0)

        input_shape = (1, 1, self.observation_size)
        data_input = data.reshape(input_shape)
        mem_input = mem.reshape(1, self.h_size)
        carry_input = carry.reshape(1, self.h_size)
        model_input = [mem_input, carry_input, data_input]

        pred = self.model.predict(model_input, batch_size = 1)
        rnd_type = np.random.randint(3)
        rnd_grid = np.random.randint(self.q_size)
        rnd_bus = np.zeros((2, obs.dim_topo))
        rnd_bus[0][obs.topo_vect == 1] = 1.0;
        rnd_bus[1][obs.topo_vect == 2] = 1.0;        
        rnd_line = np.array(obs.line_status).astype(np.float32)
        rnd_disp = np.zeros(obs.n_gen)

        # Random action type selection
        if rnd_type == 0:
            # Take random changes on topology
            rnd_bus[:] = analog.netbus_rnd(obs)[:]
        elif rnd_type == 1:
            # Switch a random line status
            rnd_line[:] = analog.netline_rnd(obs)[:]
        else:
            # Take random ramp disp
            rnd_disp[:] = analog.netdisp_rnd(obs)[:]

        return (rnd_grid, rnd_bus, rnd_line, rnd_disp), pred[4], pred[5]

    def update_target_hard(self, target_model):
        this_weights = self.model.get_weights()
        target_model.set_weights(this_weights)

    def update_target_soft(self, target_model, tau=1e-2):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.trainable_variables
        main_params = self.model.trainable_variables

        # Update each param
        for i, var in enumerate(target_params):
            var_persist = var.value() * tau_inv
            var_update = main_params[i].value() * tau
            # Poliak averaging
            var.assign(var_update + var_persist)

    def save_network(self, path):
        # Saves model at specified path as h5 file
        # nothing has changed
        self.model.save_weights(path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # nothing has changed
        self.model.load_weights(path)
        print("Succesfully loaded network from: {}".format(path))

