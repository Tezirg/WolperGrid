import os
import json
import copy
import numpy as np
import tensorflow as tf
import tree
import pandas as pd

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ReplayBuffer import ReplayBuffer
from WolperGrid_Config import WolperGrid_Config as cfg
from WolperGrid_Flann import WolperGrid_Flann
from WolperGrid_NN import WolperGrid_NN
from wg_util import *

class WolperGrid(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 action_file=None,
                 flann_index_file=None,
                 flann_pts_file=None,
                 flann_action_size=None,
                 name=__name__,
                 is_training=False):
        # Call parent constructor
        super().__init__(action_space,
                         action_space_converter=IdToAct)

        # Filter actions
        if action_file is None:
            if cfg.VERBOSE:
                print ("Filtering actions ..")
            self.action_space.filter_action(self._filter_action)
            if cfg.VERBOSE:
                print (".. Done filtering actions")
        else:
            if cfg.VERBOSE:
                print ("Loading actions ..")
            self.action_space.init_converter(all_actions=action_file)
            if cfg.VERBOSE:
                print (".. Done loading actions")

        # Store constructor params
        self.observation_space = observation_space
        self.obs_space = observation_space
        self.observation_size = wg_size_obs(self.obs_space)
        print("observation_size = ", self.observation_size)
        self.name = name
        self.batch_size = cfg.BATCH_SIZE
        self.is_training = is_training

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []

        # Create G(x)
        if flann_index_file is None or \
           flann_pts_file is None or \
           flann_action_size is None:
            self.flann = WolperGrid_Flann(self.action_space)
            self.flann.construct_flann()
        else:
            self.flann = WolperGrid_Flann(self.action_space,
                                          action_size=flann_action_size)
            self.flann.load_flann(flann_index_file, flann_pts_file)

        # Load network graph
        self.Qmain = WolperGrid_NN(self.action_space,
                                   self.observation_size,
                                   self.flann.action_size,
                                   is_training = self.is_training)

        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
        self.done = False
        self.steps = 0
        self.plays = 0
        self.last_update = -1
        self.epoch_rewards = []
        self.epoch_alive = []
        self.epoch_illegal = []
        self.epoch_ambiguous = []
        self.epoch_do_nothing = []
        self.epoch_actions = []
        self.epsilon = cfg.INITIAL_EPSILON
        self.step_epsilon = (cfg.INITIAL_EPSILON - cfg.FINAL_EPSILON)
        self.step_epsilon /= cfg.DECAY_EPSILON
        self.loss_actor = 42.0
        self.loss_critic = 42.0
        self.Qtarget = WolperGrid_NN(self.action_space,
                                     self.observation_size,
                                     self.flann.action_size,
                                     is_training = self.is_training)
        WolperGrid_NN.update_target_hard(self.Qmain.obs,
                                         self.Qtarget.obs)
        WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                         self.Qtarget.actor)
        WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                         self.Qtarget.critic)

        self.impact_tree = None
        if cfg.UNIFORM_EPSILON:
            self.impact_tree = unitary_acts_to_impact_tree(self.action_space)

    def _filter_action(self, action):
        act_dict = action.impact_on_objects()

        # Filter based on action type

        ## Filter out set_bus actions if disabled
        if not cfg.ACTION_SET_BUS: 
            if len(act_dict["topology"]["assigned_bus"]) > 0:
                return False
            if len(act_dict["topology"]["disconnect_bus"]) > 0:
                return False
        ## Filter out set_line actions if disabled
        if not cfg.ACTION_SET_LINE:
            if act_dict["force_line"]["reconnections"]["count"] > 0:
                return False
            if act_dict["force_line"]["disconnections"]["count"] > 0:
                return False
        ## Filter out change bus actions if disabled
        if not cfg.ACTION_CHANGE_BUS:
            if len(act_dict["topology"]["bus_switch"]) > 0:
                return False
        ## Filter out change line actions if disabled
        if not cfg.ACTION_CHANGE_LINE:
            if act_dict["switch_line"]["count"] > 0:
                return False
        ## Filter out redispatch actions if disabled
        if not cfg.ACTION_REDISP:
            if act_dict["redispatch"]["changed"]:
                return False

        # Custom subs exclude
        exclude_subs = [48]
        if len(exclude_subs):
            if len(act_dict["topology"]["bus_switch"]) > 0:
                for d in act_dict["topology"]["bus_switch"]:
                    if d["substation"] in exclude_subs:
                        return False
            if len(act_dict["topology"]["assigned_bus"]) > 0:
                for d in act_dict["topology"]["assigned_bus"]:
                    if d["substation"] in exclude_subs:
                        return False
            if len(act_dict["topology"]["disconnect_bus"]) > 0:
                for d in act_dict["topology"]["disconnect_bus"]:
                    if d["substation"] in exclude_subs:
                        return False

        # Custom subs select
        only_subs = []
        # only_subs = [10,11,12,15,16,29,112,30,31,113,114,26] # R1
        if len(only_subs):
            if len(act_dict["topology"]["bus_switch"]) > 0:
                for d in act_dict["topology"]["bus_switch"]:
                    if d["substation"] in only_subs:
                        return True
            elif len(act_dict["topology"]["assigned_bus"]) > 0:
                for d in act_dict["topology"]["assigned_bus"]:
                    if d["substation"] in only_subs:
                        return True
            elif len(act_dict["topology"]["disconnect_bus"]) > 0:
                for d in act_dict["topology"]["disconnect_bus"]:
                    if d["substation"] in only_subs:
                        return True
            else:
                return False

        # Not filtered, keep it
        return True

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _save_hyperparameters(self, logpath, env, iters):
        r_instance = env.reward_helper.template_reward
        hp = {
            "episodes": iters,
            "lr_actor": cfg.LR_ACTOR,
            "lr_critic": cfg.LR_CRITIC,
            "batch_size": cfg.BATCH_SIZE,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "buffer_size": cfg.REPLAY_BUFFER_SIZE,
            "buffer_min": cfg.REPLAY_BUFFER_MIN,
            "update_freq": cfg.UPDATE_FREQ,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "save_freq": cfg.SAVE_FREQ,
            "input_bias": cfg.INPUT_BIAS,
            "k": cfg.K,
            "reward": dict(r_instance)
        }
        hp_filename = "{}-hypers.json".format(self.name)
        hp_path = os.path.join(logpath, hp_filename)
        with open(hp_path, 'w') as fp:
            json.dump(hp, fp=fp, indent=2)

    ## Agent Interface
    def convert_obs(self, observation):
        return wg_convert_obs(observation)

    def convert_act(self, act_idx):
        return super().convert_act(act_idx)

    def reset(self, observation):
        self._reset_state(observation)

    def my_act(self, state, reward, done=False):
        act_idx = self.predict_move(state)

        return act_idx

    def load(self, path):
        self.Qmain.load_network(path)
        if self.is_training:
            WolperGrid_NN.update_target_hard(self.Qmain.obs,
                                             self.Qtarget.obs)
            WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                             self.Qtarget.actor)
            WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                             self.Qtarget.critic)

    def save(self, path):
        self.Qmain.save_network(path)

    def train_episode(self, m, env):
        t = 0
        total_reward = 0
        episode_illegal = 0
        episode_ambiguous = 0
        episode_do_nothing = 0
        episode_actions = []

        if cfg.VERBOSE:
            scenario_path = env.chronics_handler.real_data.get_id()
            scenario_name = os.path.basename(os.path.abspath(scenario_path))
            start_episode_msg = "Episode [{:05d}] - {} - Epsilon {}"
            print(start_episode_msg.format(m, scenario_name, self.epsilon))

        # Get once for use in loop
        act_noop = env.action_space({})
        th = env.get_thermal_limit()

        # Loop for t in episode steps
        max_t = env.chronics_handler.max_timestep() - 1
        while self.done is False and t < max_t:
            if np.random.rand(1) <= self.epsilon:
                self.plays += 1
                pred = self.random_move(self.state)
            else:
                self.plays += 1
                pred = self.predict_move(self.state)

            # Convert it to a valid action
            act = self.convert_act(pred)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            # Convert observation
            new_state = self.convert_obs(new_obs)

            if cfg.ILLEGAL_GAME_OVER and info["is_illegal"]:
                self.done = True

            # Save to exp buffer
            self.exp_buffer.add(self.state, pred,
                                reward, self.done, new_state)

            # Minibatch train
            if self.exp_buffer.size() >= cfg.REPLAY_BUFFER_MIN and \
               self.plays > self.last_update and \
               self.plays % cfg.UPDATE_FREQ == 0:
                self.last_update = self.plays
                # Sample from experience buffer
                batch = self.exp_buffer.sample(self.batch_size)
                # Perform training
                self._ddpg_train(batch, self.steps)
                # Update target network towards primary network
                if cfg.UPDATE_TARGET_SOFT_TAU > 0:
                    tau = cfg.UPDATE_TARGET_SOFT_TAU
                    WolperGrid_NN.update_target_soft(self.Qmain.obs,
                                                     self.Qtarget.obs, tau)
                    WolperGrid_NN.update_target_soft(self.Qmain.actor,
                                                     self.Qtarget.actor, tau)
                    WolperGrid_NN.update_target_soft(self.Qmain.critic,
                                                     self.Qtarget.critic, tau)
                # Update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   (t % cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    WolperGrid_NN.update_target_hard(self.Qmain.obs,
                                                     self.Qtarget.obs)
                    WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                                     self.Qtarget.actor)
                    WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                                     self.Qtarget.critic)
                # Log to tensorboard
                if self.plays % cfg.LOG_FREQ == 0:
                    self._tf_log_summary(self.steps)

            # Increment step
            if info["is_illegal"] or \
               info["is_illegal_reco"] or \
               info["is_dispatching_illegal"]:
                episode_illegal += 1
            if info["is_ambiguous"]:
                episode_ambiguous += 1
            if pred == 0:
                episode_do_nothing += 1
            episode_actions.append(pred)
            t += 1
            self.steps += 1
            total_reward += reward
            self.obs = new_obs
            self.state = new_state

        # After episode
        self.epoch_rewards.append(total_reward)
        self.epoch_alive.append(t)
        self.epoch_illegal.append(episode_illegal)
        self.epoch_ambiguous.append(episode_ambiguous)
        self.epoch_do_nothing.append(episode_do_nothing)
        self.epoch_actions.append(len(list(set(episode_actions))))
        if cfg.VERBOSE:
            done_episode_msg = "Episode [{}-{:04d}] - Steps [{}] - Reward [{}]"
            print(done_episode_msg.format(env.name, m, t, total_reward))
        # Ensure arrays dont grow too much
        if len(self.epoch_rewards) > 2048:
            self.epoch_rewards = self.epoch_rewards[-2048:]
            self.epoch_alive = self.epoch_alive[-2048:]
            self.epoch_illegal = self.epoch_illegal[-2048:]
            self.epoch_ambiguous = self.epoch_ambiguous[-2048:]
            self.epoch_do_nothing = self.epoch_do_nothing[-2048:]
            self.epoch_actions = self.epoch_actions[-2048:]

    ## Training Procedure
    def train(self, env,
              iterations=100,
              save_path="./models",
              logdir="./logs-train"):

        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name)
        os.makedirs(modelpath, exist_ok=True)
        self.tf_writer = tf.summary.create_file_writer(logpath, name=self.name)
        self._save_hyperparameters(save_path, env, iterations)

        # Save actions and flann index
        self.action_space.save(modelpath, "actions.npy")
        flann_index_path = os.path.join(modelpath, "flann.index")
        flann_pts_path = os.path.join(modelpath, "flann.npy")
        self.flann.save_flann(flann_index_path, flann_pts_path)
        
        # Training loop, over M episodes
        for m in range(iterations):
            init_obs = env.reset() # This shouldn't raise
            self.reset(init_obs)

            # Enter episode
            self.train_episode(m, env)

            # Slowly decay e-greedy rate
            if self.epsilon > cfg.FINAL_EPSILON:
                self.epsilon -= self.step_epsilon
                # Last decay may overshoot
                if self.epsilon < cfg.FINAL_EPSILON:
                    self.epsilon = cfg.FINAL_EPSILON

            # Save the network weights sometimes
            if m > 0 and m % cfg.SAVE_FREQ == 0:
                self.save(modelpath)

        # Save model after all iterations
        self.save(modelpath)

    def _tf_log_summary(self, step):
        if cfg.VERBOSE:
            print("loss actor = ", self.loss_actor)
            print("loss critic = ", self.loss_critic)
            print("exp buffer size = ", self.exp_buffer.size())
        with self.tf_writer.as_default():
            mean_reward = np.mean(self.epoch_rewards)
            mean_alive = np.mean(self.epoch_alive)
            mean_illegal = np.mean(self.epoch_illegal)
            mean_ambiguous = np.mean(self.epoch_ambiguous)
            mean_dn = np.mean(self.epoch_do_nothing)
            mean_act = np.mean(self.epoch_actions)
            mean_reward_10 = mean_reward
            mean_alive_10 = mean_alive
            mean_illegal_10 = mean_illegal
            mean_ambiguous_10 = mean_ambiguous
            mean_dn_10 = mean_dn
            mean_act_10 = mean_act
            mean_reward_100 = mean_reward
            mean_alive_100 = mean_alive
            mean_illegal_100 = mean_illegal
            mean_ambiguous_100 = mean_ambiguous
            mean_dn_100 = mean_dn
            mean_act_100 = mean_act
            if len(self.epoch_rewards) >= 10:
                mean_reward_10 = np.mean(self.epoch_rewards[-10:])
                mean_alive_10 = np.mean(self.epoch_alive[-10:])
                mean_illegal_10 = np.mean(self.epoch_illegal[-10:])
                mean_ambiguous_10 = np.mean(self.epoch_ambiguous[-10:])
                mean_dn_10 = np.mean(self.epoch_do_nothing[-10:])
                mean_act_10 = np.mean(self.epoch_actions[-10:])
            if len(self.epoch_rewards) >= 100:
                mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                mean_alive_100 = np.mean(self.epoch_alive[-100:])
                mean_illegal_100 = np.mean(self.epoch_illegal[-100:])
                mean_ambiguous_100 = np.mean(self.epoch_ambiguous[-100:])
                mean_dn_100 = np.mean(self.epoch_do_nothing[-100:])
                mean_act_100 = np.mean(self.epoch_actions[-100:])
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
            tf.summary.scalar("mean_donothing_100", mean_dn_100, step)
            tf.summary.scalar("mean_donothing_10", mean_dn_10, step)
            tf.summary.scalar("mean_act_100", mean_act_100, step)
            tf.summary.scalar("mean_act_10", mean_act_10, step)
            tf.summary.scalar("epsilon", self.epsilon, step)
            tf.summary.scalar("loss_actor", self.loss_actor, step)
            tf.summary.scalar("loss_critic", self.loss_critic, step)

    def predict_k(self, obs_input, proto,
                  use_target=False):
        # Get k actions
        k_acts = self.flann.search_flann(proto, cfg.K)
        # Get Q values
        Q = np.full(cfg.K, -1e3)

        # Select model main or target
        critic_nn = self.Qmain.critic
        if use_target:
            critic_nn = self.Qtarget.critic

        # Special handling for k = 1
        if cfg.K == 1:
            input_obs = np.repeat(obs_input, 1, axis=0)
            input_act = np.array([self.flann[k_acts[0]]])
            input_batch = [input_obs, input_act]
            Q_batch = critic_nn(input_batch, training=False)
            Q = Q_batch.numpy()[0]
            return k_acts, Q
            
        # By chunks
        k_batch = 32
        k_rest = cfg.K % k_batch
        k_batch_data = np.repeat(obs_input, k_batch, axis=0)
        for i in range (0, cfg.K, k_batch):
            a_s = i
            a_e = i + k_batch
            if a_e > cfg.K:
                a_e = i + k_rest
                k_batch_data = k_batch_data[:k_rest]

            act_idx_batch = k_acts[0][a_s:a_e]
            act_batch_li = [self.flann[a] for a in act_idx_batch]
            act_batch = np.array(act_batch_li)
            input_batch = [k_batch_data, act_batch]
            Q_batch = critic_nn(input_batch, training=False)
            Q_batch = Q_batch.numpy().reshape(a_e - a_s)
            Q[a_s:a_e] = Q_batch[:]

        return k_acts[0], Q

    def predict_move(self, data,
                     use_target=False,
                     batch_dbg=-1):
        input_shape = (1, self.observation_size)
        data_input = data.reshape(input_shape)
        obs_input = [data_input]
        if use_target:
            obs = self.Qtarget.obs(obs_input, training=False)
            proto = self.Qtarget.actor(obs, training=False)
        else:
            obs = self.Qmain.obs(obs_input, training=False)
            proto = self.Qmain.actor(obs, training=False)

        # Send k closest actions to critic
        k_acts, Q = self.predict_k(obs.numpy(),
                                   proto.numpy(),
                                   use_target)

        # Get index of highest critic q value out of K
        k_index = np.argmax(Q)

        # Get action index
        act_index = k_acts[k_index]

        # Simulate if enabled
        cfg_sim_enabled = (cfg.SIMULATE > 0) or (cfg.SIMULATE_DO_NOTHING)
        sim_enabled = (not use_target) and cfg_sim_enabled
        if sim_enabled:
            # Simulate Top Q
            act = self.convert_act(act_index)
            _, r, _, _ = self.obs.simulate(act)

            if cfg.SIMULATE > 0:
                # Simulate top critic [cfg.SIMULATE] Q values
                # Keep the index of the highest simulate result
                k_indexes = np.argpartition(Q, -cfg.SIMULATE)[-cfg.SIMULATE:]
                # Skip Top Q, already simulated
                for k_idx in k_indexes[:-1]:
                    act_idx = k_acts[k_idx]
                    act = self.convert_act(act_idx)
                    _, r_test, _, _ = self.obs.simulate(act)
                    if r_test > r:
                        r = r_test
                        k_index = k_idx
                        act_index = act_idx

            if cfg.SIMULATE_DO_NOTHING:
                # Test against do nothing as well
                act = self.convert_act(0)
                _, r_test, _, _ = self.obs.simulate(act)
                if r_test > r:
                    act_index = 0

        if batch_dbg == 0 and self.plays % (cfg.UPDATE_FREQ * 100) == 0:
            pd_dbg = pd.DataFrame(np.array([
                proto.numpy()[0],
                self.flann[act_index],
            ]),
            index=[
                "proto_{}".format(self.steps),
                "flann_{}".format(self.steps)
            ])
            pd_dbg.to_csv('ddpg_dbg.csv', mode='a', header=False)

        return act_index

    def random_move(self, data):
        if cfg.UNIFORM_EPSILON:
            # Random impact
            action_class_p = [1.0/3.0,1.0/3.0,1.0/3.0]
            return draw_from_tree(self.impact_tree, action_class_p)
        else:
            # Random action
            return np.random.randint(self.action_space.n)

    def _ddpg_train(self, batch, step):
        grad_clip = 1.0
        input_shape = (self.batch_size,
                       self.observation_size)
        # S(t)
        t_data = np.vstack(batch[0])
        t_data = t_data.reshape(input_shape)
        t_a = np.array([self.flann[a] for a in batch[1]])

        # S(t+1)
        t1_data = np.vstack(batch[4])
        t1_data = t1_data.reshape(input_shape)
        t1_a = []
        for i in range(self.batch_size):
            data = np.array(batch[4][i])
            a = self.predict_move(data, use_target=True, batch_dbg=i)
            t1_a.append(self.flann[a])
        t1_a = np.array(t1_a)

        # Perform DDPG update as per DeepMind implementation:
        # github.com/deepmind/acme/blob/master/acme/agents/tf/ddpg/learning.py
        with tf.GradientTape(persistent=True) as tape:
            t_O = self.Qmain.obs([t_data])
            t1_O = self.Qtarget.obs([t1_data])
            t1_O = tree.map_structure(tf.stop_gradient, t1_O)

            t_Q = self.Qmain.critic([t_O, t_a])
            t1_Q = self.Qtarget.critic([t1_O, t1_a])

            # Flatten to [batch_size]
            t_Q = tf.squeeze(t_Q, axis=-1)
            t1_Q = tf.squeeze(t1_Q, axis=-1)

            # Critic loss / squared td error
            d = (1.0 - batch[3].astype(np.float32)) * cfg.DISCOUNT_FACTOR
            td_v = tf.stop_gradient(batch[2] + d * t1_Q)
            td_err = td_v - t_Q
            loss_c = 0.5 * tf.square(td_err)
            loss_critic = tf.math.reduce_mean(loss_c)

            # Stop gradient on obs net for policy
            dpg_t_O = tf.stop_gradient(t_O)
            dpg_t_a = self.Qmain.actor([dpg_t_O])
            dpg_t_q = self.Qmain.critic([dpg_t_O, dpg_t_a])

            # DPG / gradient sample
            # github.com/deepmind/acme/../acme/tf/losses/dpg.py

            # Don't record gradient compute on tape
            with tape.stop_recording():
                dqda = tape.gradient([dpg_t_q], [dpg_t_a])[0]

            # Invert gradient if enabled
            if cfg.GRADIENT_INVERT:
                pdiff_max = 0.5 * (-dpg_t_a + 1.0)
                pdiff_min = 0.5 * (dpg_t_a + 1.0)
                dqda_filter = tf.zeros(dpg_t_a.shape[1])
                dqda = tf.where(tf.greater(dqda, dqda_filter),
                                tf.multiply(dqda, pdiff_max),
                                tf.multiply(dqda, pdiff_min))
            
            # Gradient clip if enabled
            if cfg.GRADIENT_CLIP:
                dqda = tf.clip_by_norm(dqda, 1.0, axes=-1)
            # target
            target_a = dqda + dpg_t_a
            # Dont propagate gradient to critic/obs
            target_a = tf.stop_gradient(target_a)
            target_sq = tf.square(target_a - dpg_t_a)
            loss_act = 0.5 * tf.reduce_sum(target_sq, axis=-1)
            loss_actor = tf.reduce_mean(loss_act)

            # github.com/openai/baselines/../baselines/ddpg/ddpg_learner.py
            #loss_actor = -tf.reduce_mean(dpg_t_q)

        # Get vars
        actor_vars = self.Qmain.actor.trainable_variables
        crit_vars = (
            self.Qmain.obs.trainable_variables +
            self.Qmain.critic.trainable_variables
        )

        # Gradients
        actor_grads = tape.gradient(loss_actor, actor_vars)
        crit_grads = tape.gradient(loss_critic, crit_vars)

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Gradient clip if enabled
        if cfg.GRADIENT_CLIP:
            actor_grads = [tf.clip_by_norm(grad, grad_clip) for grad in actor_grads]
            crit_grads = [tf.clip_by_norm(grad, grad_clip) for grad in crit_grads] 

        if self.plays % (cfg.UPDATE_FREQ * 100) == 0:
            pd_dbg = pd.DataFrame(np.array([
                #dqda[0].numpy(),
                dpg_t_a[0].numpy(),
                actor_grads[-1].numpy()
            ]),
            index=[
                #"dqda_{}".format(self.steps),
                "dpg_t_a_{}".format(self.steps),
                "grads_{}".format(self.steps)
            ])
            pd_dbg.to_csv('ddpg_dbg.csv', mode='a', header=False)

        self.Qmain.actor_opt.apply_gradients(zip(actor_grads, actor_vars))
        self.Qmain.critic_opt.apply_gradients(zip(crit_grads, crit_vars))

        self.loss_critic = loss_critic.numpy()
        self.loss_actor = loss_actor.numpy()
        
