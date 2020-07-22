import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
import tensorflow.keras.models as tfkm
import tensorflow.keras.optimizers as tfko
import tensorflow.keras.layers as tfkl
import tensorflow.keras.activations as tfka
import tensorflow.keras.initializers as tfki

from WolperGrid_Config import WolperGrid_Config as cfg

uniform_initializer = tfki.VarianceScaling(
    distribution='uniform',
    mode='fan_out',
    scale=0.333)

class WolperGrid_NN(object):
    def __init__(self,
                 gridobj,
                 observation_size,
                 proto_size,
                 is_training = False):
        self.input_size = observation_size
        self.topo_size = gridobj.dim_topo
        self.n_line = gridobj.n_line
        self.disp_size = gridobj.n_gen
        self.is_training = is_training
        self.proto_size = proto_size

        self.obs_nn = False
        if self.obs_nn == True:
            self.obs_size = 2048
        else:
            self.obs_size = self.input_size

        # AC models
        self.obs = None
        self.actor = None
        self.critic = None
        self.construct_wg_obs()
        self.construct_wg_actor()
        self.construct_wg_critic()

    def construct_mlp(self,
                      layer_in,
                      layer_sizes,
                      name="mlp",
                      batch_norm=True,
                      activation=tf.nn.elu,
                      activation_final=None):
        size_index = 0
        if batch_norm:
            pre_name = "{}-bn-fc".format(name)
            layer = tfkl.Dense(layer_sizes[0], name=pre_name)(layer_in)
            bn_name = "{}-bn".format(name)
            layer = tfkl.BatchNormalization(trainable=self.is_training,
                                            name=bn_name)(layer)
            th_name = "{}-tanh".format(name)
            layer = tf.nn.tanh(layer, name=th_name)
            size_index = 1
        else:
            layer = layer_in

        for i, size in enumerate(layer_sizes[size_index:-1]):
            layer_name = "{}-fc-{}".format(name, i + 1)
            layer = tfkl.Dense(size,
                               kernel_initializer=uniform_initializer,
                               name=layer_name)(layer)
            # Add activation if provided
            if activation is not None:
                activation_name = "{}-act-{}".format(name, i + 1)
                layer = activation(layer, name=activation_name)

        # Final layer
        layer_name = "{}-fc-{}".format(name, "final")
        layer_final = tfkl.Dense(layer_sizes[-1],
                                 kernel_initializer=uniform_initializer,
                                 name=layer_name)(layer)
        if activation_final is not None:
            activation_name = "{}-act-{}".format(name, "final")
            layer_final = activation_final(layer_final, name=activation_name)

        # Return final
        return layer_final            
        
    def construct_wg_obs(self):
        input_shape = (self.input_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_shape,
                              name='input_obs')

        if self.obs_nn:
            layer_n = 8
            layer_idxs = np.arange(layer_n)
            layer_range = [0, layer_n - 1]
            size_range = [
                self.input_size,
                self.obs_size
            ]
            sizes_np = np.interp(layer_idxs, layer_range, size_range)
            sizes = list(sizes_np)
            output_obs = self.construct_mlp(input_obs,
                                            sizes,
                                            name="obs",
                                            activation=tf.nn.elu,
                                            activation_final=tf.nn.elu)
        else:
            output_obs = tfka.linear(input_obs)

        obs_inputs = [input_obs]
        obs_outputs = [output_obs]
        self.obs = tfk.Model(inputs=obs_inputs,
                             outputs=obs_outputs,
                             name="obs_"  + self.__class__.__name__)
        self.obs_opt = tfko.Adam(lr=cfg.LR_CRITIC)
        self.obs.compile(loss="mse", optimizer=self.obs_opt)

    def construct_wg_actor(self):
        # Defines input tensors and scalars
        input_shape = (self.obs_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_shape,
                              name='actor_obs')

        # Forward encode
        layer_n = 4
        layer_idxs = np.arange(layer_n)
        layer_range = [0, layer_n - 1]
        size_range = [
            self.obs_size,
            self.proto_size
        ]
        sizes_np = np.interp(layer_idxs, layer_range, size_range)
        sizes = list(sizes_np)
        proto = self.construct_mlp(input_obs,
                                   sizes,
                                   name="actor-mlp",
                                   batch_norm=True,
                                   activation=tf.nn.elu,
                                   activation_final=tf.nn.tanh)
        # Backwards pass
        actor_inputs = [ input_obs ]
        actor_outputs = [ proto ]
        self.actor = tfk.Model(inputs=actor_inputs,
                               outputs=actor_outputs,
                               name="actor_" + self.__class__.__name__)
        self.actor_opt = tfko.Adam(lr=cfg.LR_ACTOR)
        self.actor.compile(loss="mse", optimizer=self.actor_opt)

    def construct_wg_critic(self):
        input_obs_shape = (self.obs_size,)
        input_obs = tfk.Input(dtype=tf.float32,
                              shape=input_obs_shape,
                              name='critic_obs')
        input_proto_shape = (self.proto_size,)
        input_proto = tfk.Input(dtype=tf.float32,
                                shape=input_proto_shape,
                                name='critic_proto')

        input_concat = tf.concat([input_obs, input_proto], axis=-1,
                                 name="actor-concat")
        # Forward pass
        layer_n = 4
        layer_idxs = np.arange(layer_n)
        layer_range = [0, layer_n - 1]
        size_range = [
            self.obs_size + self.proto_size,
            1
        ]
        sizes_np = np.interp(layer_idxs, layer_range, size_range)
        sizes = list(sizes_np)
        Q = self.construct_mlp(input_concat,
                               sizes,
                               name="critic-mlp",
                               batch_norm=True,
                               activation=tf.nn.elu,
                               activation_final=None)
        # Backwards pass
        critic_inputs = [ input_obs, input_proto ]
        critic_outputs = [ Q ]
        self.critic = tfk.Model(inputs=critic_inputs,
                                outputs=critic_outputs,
                                name="critic_" + self.__class__.__name__)
        # Keras model
        self.critic_opt = tfko.Adam(lr=cfg.LR_CRITIC)
        self.critic.compile(loss="mse", optimizer=self.critic_opt)

    @staticmethod
    def update_target_hard(source_model, target_model):
        # Get parameters to update
        target_params = target_model.variables
        source_params = source_model.variables

        # Update each param
        for src, dest in zip(source_params, target_params):
            dest.assign(src)

    @staticmethod
    def update_target_soft(source_model, target_model, tau=1e-3):
        tau_inv = 1.0 - tau
        # Get parameters to update
        target_params = target_model.variables
        source_params = source_model.variables

        # Update each param
        for src, dest in zip(source_params, target_params):
            var_update = src.value() * tau
            var_persist = dest.value() * tau_inv
            # Polyak averaging
            dest.assign(var_update + var_persist)

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

        self.obs.load_weights(obs_path)
        self.actor.load_weights(actor_path)
        self.critic.load_weights(critic_path)
        print("Succesfully loaded network from: {}".format(path))
