import numpy as np
import pyflann as pf
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka

from wg_util import *

class WolperGrid_NN(object):
    def __init__(self,
                 observation_size,
                 topology_size,
                 line_size,
                 dispatch_size,
                 n_bus = 2,
                 k = 1,
                 learning_rate = 1e-5,
                 is_training = False):
        self.observation_size = observation_size
        self.topo_size = topology_size
        self.n_line = line_size
        self.disp_size = dispatch_size
        self.k = k
        self.n_bus = n_bus
        self.lr = learning_rate
        self.is_training = is_training

        # Inner NN sizes
        self.encoded_size = 384

        self.actor = None
        self.critic = None
        self.construct_wg_actor()
        self.construct_wg_critic()

    def forward_encode(self, inputobs, name):
        # Bayesian NN simulate using dropout
        layd = tfkl.Dropout(self.dropout_rate, name=name+"_bnn")(inputobs)

        # Three layers encoder
        lay1 = tfkl.Dense(self.encoded_size + 64, name=name+"_fc1")(layd)
        lay1 = tf.nn.leaky_relu(lay1, alpha=0.01, name=name+"_leak_fc1")

        lay2 = tfkl.Dense(self.encoded_size + 32, name=name+"_fc2")(lay1)
        lay2 = tf.nn.leaky_relu(lay2, alpha=0.01, name=name+"_leak_fc2")

        lay3 = tfkl.Dense(self.encoded_size, name=name+"_fc3")(lay2)
        lay3 = tf.nn.leaky_relu(lay3, alpha=0.01, name=name+"_leak_fc3")

        # Reshape encoded to (batch_size, trace_len, encoded_size)
        encoded_shape = (batch_size, trace_size, self.encoded_size)
        encoded = tf.reshape(lay3, encoded_shape, name=name+"_encoded")
        return encoded

    def forward_vec(self, hidden, out_size, name):
        vec_1 = tfkl.Dense(out_size + 64, name=name+"_fc1_vec")(hidden)
        vec_1 = tf.nn.leaky_relu(vec_1, alpha=0.01,
                                 name=name+"_leak1_vec")
        vec_2 = tfkl.Dense(out_size + 32,
                           name=name+"_fc2_vec")(vec_1)
        vec_2 = tf.nn.leaky_relu(vec_2, alpha=0.01,
                                 name=name+"_leak2_vec")
        vec = tfkl.Dense(out_size, name=name+"_fc3_vec")(vec_2)
        return vec

    def construct_wg_actor(self):
        # Defines input tensors and scalars
        input_shape = (self.observation_size,)
        input_layer = tfk.Input(dtype=tf.float32, shape=input_shape,
                                name='actor_obs')

        # Forward encode
        hidden = self.forward_encode(input_layer, "actor_encode")

        # Topology buses
        t_vec = self.forward_vec(hidden, self.topo_size, "actor_bus")
        # To action range [0;1;2]
        t = tf.math.sigmoid(t_vec, name="actor_bus_sig")
        t = tf.multiply(t, float(self.n_bus))
        print ("t shape=", t.shape)

        # Lines
        l_vec = self.forward_vec(hidden, self.n_line, "actor_line")
        # To action range [-1;0;1]
        l = tf.nn.tanh(l_vec, name="actor_line_tanh")
        print("l shape =", l.shape)

        # Redispatch
        d_vec = self.forward_vec(hidden, self.disp_size, "actor_disp")
        # To action range [-1;1]
        d = tf.nn.tanh(d_vec, name="actro_disp_tanh")
        print ("d shape=", d.shape)

        # Backwards pass
        actor_inputs = [ input_layer ]
        actor_outputs = [ t, l, d ]
        self.actor = tfk.Model(inputs=actor_inputs,
                               outputs=actor_outputs,
                               name="actor_" + self.__class__.__name__)
        losses = [ self._clipped_mse_loss ]

        self.act_opt = tfko.Adam(lr=self.lr, clipnorm=1.0)
        self.actor.compile(loss=losses, optimizer=self.act_opt)

    def construct_wg_critic(self):
        input_obs_shape = (self.observation_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_obs_shape,
                              name='critic_obs')
        input_proto_shape = (self.topo_size + self.n_line + self.disp_size,)
        input_proto = tfk.Input(dtype=tf.float32,
                                shape=input_proto_shape,
                                name='critic_proto')

        # Forward encode
        encoded_obs = self.forward_encode(input_obs, "critic_obs")
        encoded_act = self.forward_encode(input_proto, "critic_act")

        hidden = tf.concat([encoded_obs, encoded_act], "critic_concat")

        # Q values for K closest actions
        kQ = self.forward_vec(hidden, self.k)

        # Backwards pass
        critic_inputs = [ input_layer ]
        critic_outputs = [ kQ ]
        self.critic = tfk.Model(inputs=critic_inputs,
                                outputs=critic_outputs,
                                name="critic_" + self.__class__.__name__)
        # Keras model
        self.critic_opt = tfko.Adam(lr=self.lr, clipnorm=1.0)
        self.critic.compile(loss=losses, optimizer=self.critic_opt)        

    def _no_loss(self, y_true, y_pred):
        return 0.0

    def _clipped_mse_loss(self, y_true, y_pred):
        loss = tf.math.reduce_mean(tf.math.square(y_true - y_pred),
                                   name="loss_mse")
        clipped_loss = tf.clip_by_value(loss, 0.0, 1e2, name="loss_clip")
        return clipped_loss

    def predict_move(self, data, mem, carry):
        input_shape = (1, self.observation_size)
        data_input = data.reshape(input_shape)
        model_input = [data_input]

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

    @staticmethod
    def update_target_hard(source_model, target_model):
        src_weights = source_model.get_weights()
        target_model.set_weights(src_weights)

    @staticmethod
    def update_target_soft(source_model, target_model, tau=1e-3):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.trainable_variables
        source_params = source_model.trainable_variables

        # Update each param
        for i, var in enumerate(target_params):
            var_update = source_params[i].value() * tau
            var_persist = var.value() * tau_inv
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
