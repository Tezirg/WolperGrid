# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

import os
import json
import copy
import numpy as np
import tensorflow as tf

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import AnalogStateConverter

from ReplayBuffer import ReplayBuffer
from WolperGrid_NN import WolperGrid_NN
from wg_util import *

INITIAL_EPSILON = 0.90
FINAL_EPSILON = 0.0
DECAY_EPSILON = 1024*64
STEP_EPSILON = (INITIAL_EPSILON-FINAL_EPSILON)/DECAY_EPSILON
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = 1024*4
UPDATE_TARGET_HARD_FREQ = -1
UPDATE_TARGET_SOFT_TAU = 0.001
INPUT_BIAS = 0.0
SUFFLE_FREQ = 1000
SAVE_FREQ = 100

class WolperGrid(BaseAgent):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 batch_size=1,
                 is_training=False,
                 lr=1e-5):
        # Call parent constructor
        super().__init__(action_space)

        # Store constructor params
        self.observation_space = observation_space
        self.obs_space = observation_space
        self.name = name
        self.n_grid = n_grid
        self.batch_size = batch_size
        self.is_training = is_training
        self.lr = lr

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []

        # Compute dimensions from intial state
        self.observation_size = wg_size_obs(self.obs_space)
        print("Observation_size = ", self.observation_size)

        # Load network graph
        self.Qmain = WolperGrid_NN(self.observation_size,
                                   self.observation_space.dim_topo,
                                   self.observation_space.n_line,
                                   self.observation_space.n_gen,
                                   learning_rate = self.lr,
                                   is_training = self.is_training)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE,
                                       self.batch_size)
        self.done = False
        self.steps = 0
        self.epoch_rewards = []
        self.epoch_alive = []
        self.epoch_illegal = []
        self.epoch_ambiguous = []
        self.episode_exp = []
        self.epsilon = INITIAL_EPSILON
        self.loss = -1.0
        self.Qtarget = WolperGrid_NN(self.observation_size,
                                     self.observation_space.dim_topo,
                                     self.observation_space.n_line,
                                     self.observation_space.n_gen,
                                     learning_rate = self.lr,
                                     is_training = self.is_training)
        WolpGrid_NN.update_target_hard(self.Qmain.actor, self.Qtarget.actor)
        WolpGrid_NN.update_target_hard(self.Qmain.critic, self.Qtarget.critic)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _register_experience(self, episode_exp):
        for exp in episode_exp:
            s = exp[0]
            a = exp[1]
            r = exp[2]
            d = exp[3]
            s2 = exp[4]
            self.exp_buffer.add(s, a, r, d, s2)

    def _save_hyperparameters(self, logpath, env, iters):
        r_instance = env.reward_helper.template_reward
        hp = {
            "episodes": iters,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "e_start": INITIAL_EPSILON,
            "e_end": FINAL_EPSILON,
            "e_decay": DECAY_EPSILON,
            "discount": DISCOUNT_FACTOR,
            "buffer_size": REPLAY_BUFFER_SIZE,
            "update_hard": UPDATE_TARGET_HARD_FREQ,
            "update_soft": UPDATE_TARGET_SOFT_TAU,
            "save_freq": SAVE_FREQ,
            "input_bias": INPUT_BIAS,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        return wg_convert_obs(observation)

    def convert_act(self, netbus, netline, netdisp):
        netstate = (netbus, netline, netdisp)
        return wg_convert_act(netstate)

    def reset(self, observation):
        self._reset_state(observation)

    def my_act(self, observation, reward, done=False):
        self.obs = observation
        state = self.convert_obs(observation)
        net_pred = self.Qmain.predict_move(state)
        bus = net_pred[0][1]
        line = net_pred[0][2]
        disp = net_pred[0][3]

        return self.convert_act(bus, line, disp)

    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    def train_episode(self, m):
        t = 0
        total_reward = 0
        episode_illegal = 0
        episode_ambiguous = 0

        print("Episode [{:04d}] - Epsilon {}".format(m, self.epsilon))
        # Loop for t in episode steps
        while self.done is False:
            # Choose an action
            if np.random.rand(1) < self.epsilon:
                pred = self.Qmain.random_move(self.state, self.obs)
            else:
                pred = self.Qmain.predict_move(self.state, self.obs)

            # Convert it to a valid action
            act = self.convert_act(pred[0][1], pred[0][2], pred[0][3])
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            # Convert observation
            new_state = self.convert_obs(new_obs)

            # Save to current episode experience
            self.episode_exp.append((self.state, pred[0], reward,
                                     self.done, new_state))

            # Minibatch train
            if self.exp_buffer.size() >= self.batch_size:
                # Sample from experience buffer
                batch = self.exp_buffer.sample()
                # Perform training
                self._batch_train(batch, self.steps)
                # Update target network towards primary network
                if UPDATE_TARGET_SOFT_TAU > 0:
                    tau = UPDATE_TARGET_SOFT_TAU
                    WolpGrid_NN.update_target_soft(self.Qmain.actor, self.Qtarget.actor, tau)
                    WolpGrid_NN.update_target_soft(self.Qmain.critic, self.Qtarget.critic, tau)
                # Update target completely
                if UPDATE_TARGET_HARD_FREQ > 0 and \
                   (self.steps % UPDATE_TARGET_HARD_FREQ) == 0:
                    WolpGrid_NN.update_target_hard(self.Qmain.actor, self.Qtarget.actor)
                    WolpGrid_NN.update_target_hard(self.Qmain.critic, self.Qtarget.critic)

            # Increment step
            if info["is_illegal"]:
                episode_illegal += 1
            if info["is_ambiguous"]:
                episode_ambiguous += 1
            t += 1
            self.steps += 1
            total_reward += reward
            self.obs = new_obs
            self.state = new_state

        # After episode
        self.epoch_rewards.append(total_reward)
        self.epoch_alive.append(alive_steps)
        self.epoch_illegal.append(episode_illegal)
        self.epoch_ambiguous.append(episode_ambiguous)
        print("Episode [{:04d}] -- Steps [{}] -- Reward [{}]".format(m, t, total_reward))

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              num_pre_training_steps = 0,
              logdir = "logs"):
        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".tf")
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, iterations)

        # Training loop, over M episodes
        for m in range(iterations):
            init_obs = env.reset() # This shouldn't raise
            self.reset(init_obs)

            # Enter episode
            self.train_episode(m)
            
            # Push last episode experience to experience buffer
            self._register_experience(self.episode_exp)
            self.episode_exp = []

            # Slowly decay e-greedy rate
            if self.epsilon > FINAL_EPSILON:
                self.epsilon -= STEP_EPSILON
            if self.epsilon < FINAL_EPSILON:
                self.epsilon = FINAL_EPSILON

            # Shuffle the data every now and then
            if m % SUFFLE_FREQ == 0:
                def shuff(x):
                    lx = len(x)
                    s = np.random.choice(lx, size=lx, replace=False)
                    return x[s]
                env.chronics_handler.shuffle(shuffler=shuff)

            # Log to tensorboard every 100 episodes
            if m % SAVE_FREQ == 0:
                self._tf_log_summary(self.loss, m)

            # Save the network every 100 episodes
            if m > 0 and m % SAVE_FREQ == 0:
                self.save(modelpath)

        # Save model after all iterations
        self.save(modelpath)

    def _tf_log_summary(self, loss, step):
        print("loss =", loss)
        with self.tf_writer.as_default():
            mean_reward = np.mean(self.epoch_rewards)
            mean_alive = np.mean(self.epoch_alive)
            mean_illegal = np.mean(self.epoch_illegal)
            mean_ambiguous = np.mean(self.epoch_ambiguous)
            mean_reward_10 = mean_reward
            mean_alive_10 = mean_alive
            mean_illegal_10 = mean_illegal
            mean_ambiguous_10 = mean_ambiguous
            mean_reward_100 = mean_reward
            mean_alive_100 = mean_alive
            mean_illegal_100 = mean_illegal
            mean_ambiguous_100 = mean_ambiguous
            if len(self.epoch_rewards) >= 10:
                mean_reward_10 = np.mean(self.epoch_rewards[-10:])
                mean_alive_10 = np.mean(self.epoch_alive[-10:])
                mean_illegal_10 = np.mean(self.epoch_illegal[-10:])
                mean_ambiguous_10 = np.mean(self.epoch_ambiguous[-10:])
            if len(self.epoch_rewards) >= 100:
                mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                mean_alive_100 = np.mean(self.epoch_alive[-100:])
                mean_illegal_100 = np.mean(self.epoch_illegal[-100:])
                mean_ambiguous_100 = np.mean(self.epoch_ambiguous[-100:])
            tf.summary.scalar("mean_reward", mean_reward, step)
            tf.summary.scalar("mean_reward_100", mean_reward_100, step)
            tf.summary.scalar("mean_reward_10", mean_reward_10, step)
            tf.summary.scalar("mean_alive", mean_alive, step)
            tf.summary.scalar("mean_alive_100", mean_alive_100, step)
            tf.summary.scalar("mean_alive_10", mean_alive_10, step)
            tf.summary.scalar("mean_illegal", mean_illegal, step)
            tf.summary.scalar("mean_illegal_100", mean_illegal_100, step)
            tf.summary.scalar("mean_illegal_10", mean_illegal_10, step)
            tf.summary.scalar("mean_ambiguous", mean_ambiguous, step)
            tf.summary.scalar("mean_ambiguous_100", mean_ambiguous_100, step)
            tf.summary.scalar("mean_ambiguous_10", mean_ambiguous_10, step)
            tf.summary.scalar("loss", loss, step)
        
    def _batch_train(self, batch, step):
        """Trains network to fit given parameters"""
        input_shape = (self.batch_size,
                       self.observation_size)
        q_data = np.vstack(batch[:, 0])
        q_data = q_data.reshape(input_shape)
        q1_data = np.vstack(batch[:, 4])
        q1_data = q1_data.reshape(input_shape)
        q_input = [q_data]
        q1_input = [q1_data]

        # Save the graph just the first time
        if step == 0:
            tf.summary.trace_on()

        # T batch predict
        pred = self.Qmain.model.predict(q_input,
                                        batch_size=self.batch_size)
        Q = pred[0]
        batch_bus = pred[1]
        batch_line = pred[2]
        batch_disp = pred[3]

        ## Log graph once and disable graph logging
        if step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        Qn, *_ = self.Qtarget.model.predict(q1_input,
                                            batch_size=self.batch_size)
        
        # Compute batch Q update to Qtarget
        for i in range(self.batch_size):
            idx = i * (self.trace_length - 1)
            a = batch[idx][1]
            grid = a[0]
            batch_bus[i][:] = a[1][:]
            batch_line[i][:] = a[2][:]
            batch_disp[i][:] = a[3][:]
            r = batch[idx][2]
            d = batch[idx][3]
            Q[i][grid] = r
            if d == False:
                Q[i][grid] += DISCOUNT_FACTOR * Qn[i][grid]

        # Batch train
        batch_x = [q_data]
        batch_y = [
            Q,
            batch_bus, batch_line, batch_disp
        ]
        loss = self.Qmain.model.train_on_batch(batch_x, batch_y)
        loss = loss[0]
        self.loss = loss
