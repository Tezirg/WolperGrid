import os
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

kernel_init1 = tfk.initializers.he_normal()
#kernel_init1 = tfk.initializers.GlorotUniform()

class WolperGrid_NN(object):
    def __init__(self,
                 observation_space,
                 action_space,
                 n_bus = 2,
                 learning_rate = 1e-5,
                 is_training = False):
        self.observation_size = wg_size_obs(observation_space)
        self.topo_size = observation_space.dim_topo
        self.n_line = observation_space.n_line
        self.disp_size = observation_space.n_gen
        self.n_bus = n_bus
        self.lr = learning_rate
        self.is_training = is_training

        # NN sizes
        self.input_size = self.observation_size
        self.obs_size = 756
        self.proto_size = self.n_line * 2 + self.topo_size * 2 + self.disp_size
        self.encoded_size = 512

        self.obs = None
        self.actor = None
        self.critic = None
        self.construct_wg_obs()
        self.construct_wg_actor()
        self.construct_wg_critic()

    def forward_encode(self, layin, size, name):
        # Multi layers encoder
        lay1 = tfkl.Dense(size + 256,
                          kernel_initializer=kernel_init1,
                          name=name+"_fc1")(layin)
        lay1 = tf.nn.elu(lay1, name=name+"_relu_fc1")

        lay2 = tfkl.Dense(size + 128,
                          kernel_initializer=kernel_init1,
                          name=name+"_fc2")(lay1)
        lay2 = tf.nn.elu(lay2, name=name+"_relu_fc2")

        lay3 = tfkl.Dense(size + 64,
                          kernel_initializer=kernel_init1,
                          name=name+"_fc3")(lay2)
        lay3 = tf.nn.elu(lay3, name=name+"_relu_fc3")
        
        lay4 = tfkl.Dense(size,
                          kernel_initializer=kernel_init1,
                          name=name+"_fc4")(lay3)

        return lay4

    def forward_vec(self, hidden, out_size, name):
        vec_1 = tfkl.Dense(out_size + 128,
                           kernel_initializer=kernel_init1,
                           name=name+"_fc1_vec")(hidden)
        vec_1 = tf.nn.elu(vec_1, name=name+"_relu1_vec")
        vec_2 = tfkl.Dense(out_size + 64,
                           kernel_initializer=kernel_init1,
                           name=name+"_fc2_vec")(vec_1)
        vec_2 = tf.nn.elu(vec_2, name=name+"_relu2_vec")
        vec = tfkl.Dense(out_size,
                         kernel_initializer=kernel_init1,
                         name=name+"_fc3_vec")(vec_2)
        return vec

    def construct_wg_actor(self):
        # Defines input tensors and scalars
        input_shape = (self.obs_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_shape,
                              name='actor_obs')

        # Forward encode
        hidden = self.forward_encode(input_obs,
                                     self.encoded_size,
                                     "actor_encode")

        # Lines
        #set_line = self.forward_vec(hidden, self.n_line,
        #                            "actor_set_line")
        #change_line = self.forward_vec(hidden, self.n_line,
        #                               "actor_change_line")
        # To set action range [-1;0;1]
        #set_line = tf.nn.tanh(set_line, name="actor_set_line_tanh")
        # To change action range [-1;1]
        #change_line = tf.nn.tanh(change_line,
        #                         name="actor_change_line_tanh")

        # Debug lines tensor shapes
        #print("set_line shape =", set_line.shape)
        #print("change_line shape =", change_line.shape)

        # Buses
        #set_bus = self.forward_vec(hidden, self.topo_size,
        #                           "actor_set_bus")
        #change_bus = self.forward_vec(hidden, self.topo_size,
        #                              "actor_change_bus")
        # To set action range [-1;0;1]
        #set_bus = tf.nn.tanh(set_bus, name="actor_set_bus_tanh")
        # To change action range [-1;1]
        #change_bus = tf.nn.tanh(change_bus,
        #                        name="actor_change_bus_tanh")

        # Debug buses tensors shapes
        #print ("set_bus shape=", set_bus.shape)
        #print ("change_bus shape=", change_bus.shape)

        # Redispatch
        #redisp = self.forward_vec(hidden, self.disp_size, "actor_redisp")
        # To action range [-1;1]
        #redisp = tf.nn.tanh(redisp, name="actor_redisp_tanh")
        
        # Debug redisp tensor shape
        #print ("redisp shape=", redisp.shape)

        # Proto action
        #proto_vects = [set_line, change_line, set_bus, change_bus, redisp]
        #proto = tf.concat(proto_vects, axis=1, name="actor_concat")
        
        proto = self.forward_vec(hidden,
                                 self.proto_size,
                                 "actor_proto1")
        #proto = tf.nn.elu(proto, name="actor_elu")
        #proto = self.forward_vec(proto, self.act_v_size, "actor_proto2")
        # To action range [-1;1]
        #proto = tf.nn.tanh(proto, name="actor_proto_tanh")
        
        # Backwards pass
        actor_inputs = [ input_obs ]
        actor_outputs = [ proto ]
        self.actor = tfk.Model(inputs=actor_inputs,
                               outputs=actor_outputs,
                               name="actor_" + self.__class__.__name__)
        losses = [ self._me_loss ]

        self.actor_opt = tfko.Adam(lr=self.lr)
        self.actor.compile(loss=losses, optimizer=self.actor_opt)

    def construct_wg_critic(self):
        input_shape = (self.obs_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_shape,
                              name='critic_obs')
        input_proto_shape = (self.proto_size,)
        input_proto = tfk.Input(dtype=tf.float32,
                                shape=input_proto_shape,
                                name='critic_proto')

        input_concat = tf.concat([input_obs, input_proto], axis=-1,
                                 name="critic_concat")

        a1 = tfkl.Dense(1024,
                        kernel_initializer=kernel_init1,
                        name="critic_linear")(input_concat)
        a2 = tf.nn.elu(a1, name="critic_relu1")
        a3 = tfkl.Dense(728,
                        kernel_initializer=kernel_init1,
                        name="critic_linear2")(a2)
        a4 = tf.nn.elu(a3, name="critic_relu2")
        a3 = self.forward_vec(a4, 384, "critic_l3")
        Q = self.forward_vec(a3, 1, "critic_Q")

        # Backwards pass
        critic_inputs = [ input_obs, input_proto ]
        critic_outputs = [ Q ]
        self.critic = tfk.Model(inputs=critic_inputs,
                                outputs=critic_outputs,
                                name="critic_" + self.__class__.__name__)

        losses = [ self._mse_loss ]
        # Keras model
        self.critic_opt = tfko.Adam(lr=self.lr)
        self.critic.compile(loss=losses, optimizer=self.critic_opt)        

    def construct_wg_obs(self):
        input_shape = (self.input_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_shape,
                              name='input_obs')
        output_obs = self.forward_encode(input_obs,
                                         self.obs_size,
                                         "obs_enc")
        obs_inputs = [input_obs]
        obs_outputs = [output_obs]
        self.obs = tfk.Model(inputs=obs_inputs,
                             outputs=obs_outputs,
                             name="obs_"  + self.__class__.__name__)
        losses = [ self._mse_loss ]
        self.obs_opt = tfko.Adam(lr=self.lr)
        self.obs.compile(loss=losses, optimizer=self.obs_opt)

    def _me_loss(self, y_true, y_pred):
        error = tf.math.abs(y_true - y_pred)
        loss = tf.math.reduce_mean(error, name="loss_me")
        return loss

    def _mse_loss(self, y_true, y_pred):
        sq_error = tf.math.square(y_true - y_pred)
        loss = tf.math.reduce_mean(sq_error,
                                   name="loss_mse")
        return loss

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
            # Polyak averaging
            var.assign(var_update + var_persist)

    def save_network(self, path):
        # Saves model at specified path

        # Compute paths
        obs_path = os.path.join(path, "obs.tf")
        actor_path = os.path.join(path, "actor.tf")
        critic_path = os.path.join(path, "critic.tf")

        self.obs.save_weights(obs_path)
        self.actor.save_weights(actor_path)
        self.critic.save_weights(critic_path)
        print("Successfully saved model at: {}".format(path))

    def load_network(self, path):
        # Compute paths
        obs_path = os.path.join(path, "obs.tf")
        actor_path = os.path.join(path, "actor.tf")
        critic_path = os.path.join(path, "critic.tf")

        self.obs.load_weights(actor_path)
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        print("Succesfully loaded network from: {}".format(path))
