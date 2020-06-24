import os
import json
import copy
import numpy as np
import tensorflow as tf

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ReplayBuffer import ReplayBuffer
from WolperGrid_Config import WolperGrid_Config as cfg
from WolperGrid_NN import WolperGrid_NN
from wg_util import *

class WolperGrid(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 is_training=False):
        # Call parent constructor
        super().__init__(action_space,
                         action_space_converter=IdToAct)

        # Store constructor params
        self.observation_space = observation_space
        self.obs_space = observation_space
        self.observation_size = wg_size_obs(self.obs_space)
        print("observation_size = ", self.observation_size)
        self.name = name
        self.batch_size = cfg.BATCH_SIZE
        self.is_training = is_training
        self.lr = cfg.LR

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []

        # Load network graph
        self.Qmain = WolperGrid_NN(self.observation_space,
                                   self.action_space,
                                   k_ratio = cfg.K_RATIO,
                                   learning_rate = cfg.LR,
                                   is_training = self.is_training)

        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ReplayBuffer(cfg.REPLAY_BUFFER_SIZE)
        self.done = False
        self.steps = 0
        self.epoch_rewards = []
        self.epoch_alive = []
        self.epoch_illegal = []
        self.epoch_ambiguous = []
        self.epoch_do_nothing = []
        self.epsilon = cfg.INITIAL_EPSILON
        self.step_epsilon = (cfg.INITIAL_EPSILON - cfg.FINAL_EPSILON)
        self.step_epsilon /= cfg.DECAY_EPSILON
        self.loss_actor = 42.0
        self.loss_critic = 42.0
        self.Qtarget = WolperGrid_NN(self.observation_space,
                                     self.action_space,
                                     k_ratio = cfg.K_RATIO,
                                     learning_rate = self.lr,
                                     is_training = self.is_training,
                                     is_target = True)
        WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                         self.Qtarget.actor)
        WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                         self.Qtarget.critic)

        self.impact_tree = None
        if cfg.UNIFORM_EPSILON:
            self.impact_tree = unitary_acts_to_impact_tree(self.action_space)

    def _reset_state(self, current_obs):
        # Initial state
        self.obs = current_obs
        self.state = self.convert_obs(self.obs)
        self.done = False

    def _save_hyperparameters(self, logpath, env, iters):
        r_instance = env.reward_helper.template_reward
        hp = {
            "episodes": iters,
            "lr": cfg.LR,
            "batch_size": cfg.BATCH_SIZE,
            "e_start": cfg.INITIAL_EPSILON,
            "e_end": cfg.FINAL_EPSILON,
            "e_decay": cfg.DECAY_EPSILON,
            "discount": cfg.DISCOUNT_FACTOR,
            "buffer_size": cfg.REPLAY_BUFFER_SIZE,
            "update_hard": cfg.UPDATE_TARGET_HARD_FREQ,
            "update_soft": cfg.UPDATE_TARGET_SOFT_TAU,
            "save_freq": cfg.SAVE_FREQ,
            "input_bias": cfg.INPUT_BIAS,
            "k": cfg.K_RATIO,
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
            self.Qmain.update_target_hard(self.Qtarget.model)

    def save(self, path):
        self.Qmain.save_network(path)

    def train_episode(self, m, env):
        t = 0
        total_reward = 0
        episode_illegal = 0
        episode_ambiguous = 0
        episode_do_nothing = 0

        if cfg.VERBOSE:
            start_episode_msg = "Episode [{:04d}] - Epsilon {}"
            print(start_episode_msg.format(m, self.epsilon))

        # Loop for t in episode steps
        max_t = env.chronics_handler.max_timestep() - 1
        while self.done is False and t < max_t:
            # Choose an action
            if np.random.rand(1) <= self.epsilon:
                pred = self.random_move(self.state)
            else:
                pred = self.predict_move(self.state)

            # Convert it to a valid action
            act = self.convert_act(pred)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            # Convert observation
            new_state = self.convert_obs(new_obs)

            # Save to exp buffer
            self.exp_buffer.add(self.state, pred,
                                reward, self.done, new_state)

            # Minibatch train
            if self.exp_buffer.size() >= self.batch_size and \
               self.steps % cfg.UPDATE_FREQ == 0:
                # Sample from experience buffer
                batch = self.exp_buffer.sample(self.batch_size)
                # Perform training
                self._ddpg_train(batch, self.steps)
                # Update target network towards primary network
                if cfg.UPDATE_TARGET_SOFT_TAU > 0:
                    tau = cfg.UPDATE_TARGET_SOFT_TAU
                    WolperGrid_NN.update_target_soft(self.Qmain.actor,
                                                     self.Qtarget.actor, tau)
                    WolperGrid_NN.update_target_soft(self.Qmain.critic,
                                                     self.Qtarget.critic, tau)
                # Update target completely
                if cfg.UPDATE_TARGET_HARD_FREQ > 0 and \
                   (t % cfg.UPDATE_TARGET_HARD_FREQ) == 0:
                    WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                                     self.Qtarget.actor)
                    WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                                     self.Qtarget.critic)
            # Log to tensorboard
            if self.steps > cfg.UPDATE_FREQ and \
               self.steps > 100 and \
               self.steps % cfg.LOG_FREQ == 0:
                self._tf_log_summary(self.steps)

            # Increment step
            if info["is_illegal"]:
                episode_illegal += 1
            if info["is_ambiguous"]:
                episode_ambiguous += 1
            if pred == 0:
                episode_do_nothing += 1
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
        if cfg.VERBOSE:
            done_episode_msg = "Episode [{:04d}] -- Steps [{}] -- Reward [{}]"
            print(done_episode_msg.format(m, t, total_reward))
        # Ensure arrays dont grow too much
        if len(self.epoch_rewards) > 1000:
            self.epoch_rewards = self.epoch_rewards[-1000:]
            self.epoch_alive = self.epoch_alive[-1000:]
            self.epoch_illegal = self.epoch_illegal[-1000:]
            self.epoch_ambiguous = self.epoch_ambiguous[-1000:]
            self.epoch_do_nothing = self.epoch_do_nothing[-1000:]

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

        # Shuffle training data
        def shuff(x):
            lx = len(x)
            s = np.random.choice(lx, size=lx, replace=False)
            return x[s]
        env.chronics_handler.shuffle(shuffler=shuff)
        
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

            # Save the network every 100 episodes
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
            mean_reward_10 = mean_reward
            mean_alive_10 = mean_alive
            mean_illegal_10 = mean_illegal
            mean_ambiguous_10 = mean_ambiguous
            mean_dn_10 = mean_dn
            mean_reward_100 = mean_reward
            mean_alive_100 = mean_alive
            mean_illegal_100 = mean_illegal
            mean_ambiguous_100 = mean_ambiguous
            mean_dn_100 = mean_dn
            if len(self.epoch_rewards) >= 10:
                mean_reward_10 = np.mean(self.epoch_rewards[-10:])
                mean_alive_10 = np.mean(self.epoch_alive[-10:])
                mean_illegal_10 = np.mean(self.epoch_illegal[-10:])
                mean_ambiguous_10 = np.mean(self.epoch_ambiguous[-10:])
                mean_dn_10 = np.mean(self.epoch_do_nothing[-10:])
            if len(self.epoch_rewards) >= 100:
                mean_reward_100 = np.mean(self.epoch_rewards[-100:])
                mean_alive_100 = np.mean(self.epoch_alive[-100:])
                mean_illegal_100 = np.mean(self.epoch_illegal[-100:])
                mean_ambiguous_100 = np.mean(self.epoch_ambiguous[-100:])
                mean_dn_100 = np.mean(self.epoch_do_nothing[-100:])
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
            tf.summary.scalar("epsilon", self.epsilon, step)
            tf.summary.scalar("loss_actor", self.loss_actor, step)
            tf.summary.scalar("loss_critic", self.loss_critic, step)

    def predict_k(self, data_input, proto, use_target=False):
        # Get k actions
        k_acts = self.Qmain.search_flann(proto)

        # Get Q values
        Q = np.zeros(self.Qmain.k)
        # By chunks
        k_batch = 512
        k_rest = self.Qmain.k % k_batch
        k_batch_data = np.repeat(data_input, k_batch, axis=0)
        for i in range (0, self.Qmain.k, k_batch):
            a_s = i
            a_e = i + k_batch
            if a_e >= self.Qmain.k:
                a_e = i + k_rest
                k_batch_data = k_batch_data[:k_rest]

            act_idx_batch = k_acts[0][a_s:a_e]
            act_batch_li = [self.Qmain.act_vects[a] for a in act_idx_batch]
            act_batch = np.array(act_batch_li)
            input_batch = [k_batch_data, act_batch]
            if use_target:
                Q_batch = self.Qtarget.critic.predict(input_batch)
            else:
                Q_batch = self.Qmain.critic.predict(input_batch)
            Q_batch = Q_batch.reshape(a_e - a_s)
            Q[a_s:a_e] = Q_batch[:]

        return k_acts[0], Q

    def predict_move(self, data, use_target=False):
        input_shape = (1, self.observation_size)
        data_input = data.reshape(input_shape)
        actor_input = [data_input]
        if use_target:
            proto = self.Qtarget.actor.predict(actor_input,batch_size=1)
        else:
            proto = self.Qmain.actor.predict(actor_input,batch_size=1)

        k_acts, Q = self.predict_k(data_input, proto, use_target)
        k_index = 0
        if not use_target and cfg.SIMULATE > 0:
            # Simulate top critic [cfg.SIMULATE] Q values
            # Keep the index of the highest simulate result
            r = float('-inf')
            k_indexes = np.argpartition(Q, -cfg.SIMULATE)[-cfg.SIMULATE:]
            for k_idx in k_indexes:
                act_idx = k_acts[k_idx]
                act = self.convert_act(act_idx)
                _, r_test, d_test, i_test = self.obs.simulate(act)
                if r_test > r and d_test is False:
                    r = r_test
                    k_index = k_idx

            if cfg.SIMULATE_DO_NOTHING:
                # Test against do nothing as well
                act = self.convert_act(0)
                _, r_test, d_test, i_test = self.obs.simulate(act)
                if r_test > r and d_test is False:
                    return 0
        else:
            # Get index of highest critic q value
            k_index = np.argmax(Q)

        # Get action index
        act_index = k_acts[k_index]

        #if self.steps > 50000:
        #    print("----")
        #    print("Action vect = ", self.Qmain.act_vects[act_index])
        #    print("Action str =", self.convert_act(act_index))
        #    print("Action proto = ", proto)
        #    print("----")
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
        grad_clip = 40.0
        input_shape = (self.batch_size,
                       self.observation_size)
        # S(t)
        t_data = np.vstack(batch[0])
        t_data = t_data.reshape(input_shape)
        t_a = np.array([self.Qmain.act_vects[a] for a in batch[1]])

        # S(t+1)
        t1_data = np.vstack(batch[4])
        t1_data = t1_data.reshape(input_shape)
        t1_a = []
        for i in range(self.batch_size):
            data = np.array(batch[4][i])
            a = self.predict_move(data, use_target=True)
            t1_a.append(self.Qmain.act_vects[a])
        t1_a = np.array(t1_a)

        # Perform DDPG update as per DeepMind implementation:
        # github.com/deepmind/acme/blob/master/acme/agents/tf/ddpg/learning.py
        with tf.GradientTape(persistent=True) as tape:
            t_Q = self.Qmain.critic([t_data, t_a])
            t1_Q = self.Qtarget.critic([t1_data, t1_a])

            # Flatten to [batch_size]
            t_Q = tf.squeeze(t_Q, axis=-1)
            t1_Q = tf.squeeze(t1_Q, axis=-1)

            # Critic loss / squared td error
            # github.com/deepmind/trfl/blob/master/trfl/value_ops.py#L34
            d = (1.0 - batch[3]) * cfg.DISCOUNT_FACTOR
            td_v = tf.stop_gradient(batch[2] + d * t1_Q)
            td_err = td_v - t_Q
            loss_c = 0.5 * tf.square(td_err)
            loss_critic = tf.math.reduce_mean(loss_c, axis=0)

            dpg_t_a = self.Qmain.actor([t_data])
            dpg_t_q = self.Qmain.critic([t_data, dpg_t_a])
            # DPG / gradient sample
            # github.com/deepmind/acme/blob/master/acme/tf/losses/dpg.py
            dqda = tape.gradient([dpg_t_q], [dpg_t_a])[0]
            # Gradient clip if enabled
            if cfg.GRADIENT_CLIP:
                dqda = tf.clip_by_norm(dqda, 1.0, axes=-1)
            target_a = dqda + dpg_t_a
            target_a = tf.stop_gradient(target_a)
            loss_actor = 0.5 * tf.reduce_sum(tf.square(target_a - dpg_t_a),
                                             axis=-1)
            loss_actor = tf.reduce_mean(loss_actor, axis=0)

        # Gradients
        actor_vars = self.Qmain.actor.trainable_variables
        crit_vars = self.Qmain.critic.trainable_variables

        actor_grads = tape.gradient(loss_actor, actor_vars)
        crit_grads = tape.gradient(loss_critic, crit_vars)

        # Delete the tape manually because of the persistent=True flag.
        del tape

        # Gradient clip if enabled
        if cfg.GRADIENT_CLIP:
            actor_grads = tf.clip_by_global_norm(actor_grads, grad_clip)[0]
            crit_grads = tf.clip_by_global_norm(crit_grads, grad_clip)[0]

        self.Qmain.actor_opt.apply_gradients(zip(actor_grads, actor_vars))
        self.Qmain.critic_opt.apply_gradients(zip(crit_grads, crit_vars))

        self.loss_critic = loss_critic.numpy()
        self.loss_actor = loss_actor.numpy()
        
