import os
import json
import copy
import numpy as np
import tensorflow as tf

from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct

from ReplayBuffer import ReplayBuffer
from WolperGrid_NN import WolperGrid_NN
from wg_util import *

INITIAL_EPSILON = 0.99
FINAL_EPSILON = 0.001
DECAY_EPSILON = 1024*2
STEP_EPSILON = (INITIAL_EPSILON-FINAL_EPSILON)/DECAY_EPSILON
DISCOUNT_FACTOR = 0.99
REPLAY_BUFFER_SIZE = 1024*8
UPDATE_TARGET_HARD_FREQ = -1
UPDATE_TARGET_SOFT_TAU = 1e-3
INPUT_BIAS = 0.0
SAVE_FREQ = 100
K_RATIO = 0.05

class WolperGrid(AgentWithConverter):
    def __init__(self,
                 observation_space,
                 action_space,
                 name=__name__,
                 batch_size=1,
                 is_training=False,
                 lr=1e-5):
        # Call parent constructor
        super().__init__(action_space,
                         action_space_converter=IdToAct)

        # Store constructor params
        self.observation_space = observation_space
        self.obs_space = observation_space
        self.observation_size = wg_size_obs(self.obs_space)
        self.name = name
        self.batch_size = batch_size
        self.is_training = is_training
        self.lr = lr

        # Declare required vars
        self.Qmain = None
        self.obs = None
        self.state = []

        # Load network graph
        self.Qmain = WolperGrid_NN(self.observation_space,
                                   self.action_space,
                                   k_ratio = K_RATIO,
                                   learning_rate = self.lr,
                                   is_training = self.is_training)
        # Setup training vars if needed
        if self.is_training:
            self._init_training()


    def _init_training(self):
        self.exp_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.done = False
        self.steps = 0
        self.epoch_rewards = []
        self.epoch_alive = []
        self.epoch_illegal = []
        self.epoch_ambiguous = []
        self.episode_exp = []
        self.epsilon = INITIAL_EPSILON
        self.loss_actor = 42.0
        self.loss_critic = 42.0
        self.Qtarget = WolperGrid_NN(self.observation_space,
                                     self.action_space,
                                     k_ratio = K_RATIO,
                                     learning_rate = self.lr,
                                     is_training = self.is_training,
                                     is_target = True)
        WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                         self.Qtarget.actor)
        WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                         self.Qtarget.critic)

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
            "k": K_RATIO,
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

    def my_act(self, observation, reward, done=False):
        self.obs = observation
        state = self.convert_obs(observation)
        act_idx = self.Qmain.predict_move(state)

        return self.convert_act(act_idx)

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

        start_episode_msg = "Episode [{:04d}] - Epsilon {}"
        print(start_episode_msg.format(m, self.epsilon))

        # Loop for t in episode steps
        max_t = env.chronics_handler.max_timestep() - 1
        while self.done is False and t < max_t:
            # Choose an action
            if np.random.rand(1) <= self.epsilon:
                pred, kpred = self.Qmain.random_move(self.state)
            else:
                pred, kpred = self.Qmain.predict_move(self.state)

            # Convert it to a valid action
            act = self.convert_act(pred)
            # Execute action
            new_obs, reward, self.done, info = env.step(act)
            # Convert observation
            new_state = self.convert_obs(new_obs)

            # Save to current episode experience
            self.episode_exp.append((self.state, (pred, kpred), reward,
                                     self.done, new_state))

            # Minibatch train
            if self.exp_buffer.size() >= self.batch_size:
                # Sample from experience buffer
                batch = self.exp_buffer.sample(self.batch_size)
                # Perform training
                self._batch_train(batch, self.steps)
                # Update target network towards primary network
                if UPDATE_TARGET_SOFT_TAU > 0:
                    tau = UPDATE_TARGET_SOFT_TAU
                    WolperGrid_NN.update_target_soft(self.Qmain.actor,
                                                     self.Qtarget.actor, tau)
                    WolperGrid_NN.update_target_soft(self.Qmain.critic,
                                                     self.Qtarget.critic, tau)
                # Update target completely
                if UPDATE_TARGET_HARD_FREQ > 0 and \
                   (self.steps % UPDATE_TARGET_HARD_FREQ) == 0:
                    WolperGrid_NN.update_target_hard(self.Qmain.actor,
                                                     self.Qtarget.actor)
                    WolperGrid_NN.update_target_hard(self.Qmain.critic,
                                                     self.Qtarget.critic)

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
        self.epoch_alive.append(t)
        self.epoch_illegal.append(episode_illegal)
        self.epoch_ambiguous.append(episode_ambiguous)
        done_episode_msg = "Episode [{:04d}] -- Steps [{}] -- Reward [{}]"
        print(done_episode_msg.format(m, t, total_reward))

    ## Training Procedure
    def train(self, env,
              iterations,
              save_path,
              logdir = "logs"):
        # Create file system related vars
        logpath = os.path.join(logdir, self.name)
        os.makedirs(save_path, exist_ok=True)
        modelpath = os.path.join(save_path, self.name + ".tf")
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
            
            # Push last episode experience to experience buffer
            self._register_experience(self.episode_exp)
            self.episode_exp = []

            # Slowly decay e-greedy rate
            if self.epsilon > FINAL_EPSILON:
                self.epsilon -= STEP_EPSILON
            if self.epsilon < FINAL_EPSILON:
                self.epsilon = FINAL_EPSILON

            # Log to tensorboard
            self._tf_log_summary(m)

            # Save the network every 100 episodes
            if m > 0 and m % SAVE_FREQ == 0:
                self.save(modelpath)

        # Save model after all iterations
        self.save(modelpath)

    def _tf_log_summary(self, step):
        print("loss actor = ", self.loss_actor)
        print("loss critic = ", self.loss_critic)
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
            tf.summary.scalar("loss_actor", self.loss_actor, step)
            tf.summary.scalar("loss_critic", self.loss_critic, step)
        
    def _batch_train(self, batch, step):
        """Trains network to fit given parameters"""
        input_shape = (self.batch_size,
                       self.observation_size)
        t_data = np.vstack(batch[0])
        t_data = t_data.reshape(input_shape)
        t1_data = np.vstack(batch[4])
        t1_data = t1_data.reshape(input_shape)
        t_input = [t_data]
        t1_input = [t1_data]

        # Save the graph just the first time
        if step == 0:
            tf.summary.trace_on()

        # T batch predict
        t_proto = self.Qmain.actor.predict(t_input,
                                           batch_size=self.batch_size)
        t_input_critic = [t_data, t_proto]
        t_kQ = self.Qmain.critic.predict(t_input_critic,
                                         batch_size=self.batch_size)

        ## Log graph once and disable graph logging
        if step == 0:
            with self.tf_writer.as_default():
                tf.summary.trace_export(self.name + "-graph", step)

        # T+1 batch predict
        t1_proto = self.Qtarget.actor.predict(t1_input,
                                              batch_size=self.batch_size)
        t1_input_critic = [t_data, t_proto]
        t1_kQ = self.Qtarget.critic.predict(t1_input_critic,
                                            batch_size=self.batch_size)
        
        # Compute batch critic / actor targets
        Qtarget = np.array(t_kQ)
        Ptarget = np.zeros_like(t_proto)
        for i in range(self.batch_size):
            a = batch[1][i][0]
            kA = batch[1][i][1]
            r = batch[2][i]
            d = 1.0 - float(batch[3][i])
            Qtarget[i][kA] = r + d * DISCOUNT_FACTOR * np.argmax(t1_kQ[i])
            Ptarget[i][:] = self.convert_act(a).to_vect()[:]

        # Batch train
        batch_x_critic = [t_data, t_proto]
        batch_y_critic = [Qtarget]
        loss_critic = self.Qmain.critic.train_on_batch(batch_x_critic,
                                                       batch_y_critic)
        batch_x_actor = [t_data]
        batch_y_actor = [Ptarget]
        loss_actor = self.Qmain.actor.train_on_batch(batch_x_actor,
                                                     batch_y_actor)        

        self.loss_critic = loss_critic
        self.loss_actor = loss_actor
