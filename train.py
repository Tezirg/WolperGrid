#!/usr/bin/env python3

import argparse
import tensorflow as tf

from grid2op.MakeEnv import make
from grid2op.Reward import *
from grid2op.Action import *
from grid2op.Parameters import Parameters

from WolperGrid import WolperGrid as WGAgent
from WolperGrid_Config import WolperGrid_Config as WGConfig
from wg_util import limit_gpu_usage

DEFAULT_NAME = "WolpGrid"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_EPISODES = 10
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-4
DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Train baseline GridRDQN")

    # Paths
    parser.add_argument("--name", required=False, default=DEFAULT_NAME,
                        help="The name of the model")
    parser.add_argument("--data_dir", required=False,
                        default="l2rpn_case14_sandbox",
                        help="Path to the dataset root directory")
    parser.add_argument("--save_dir", required=False,
                        default=DEFAULT_SAVE_DIR, type=str,
                        help="Directory where to save the model")
    parser.add_argument("--load_file", required=False,
                        help="Path to model.h5 to resume training with")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOG_DIR, type=str,
                        help="Directory to save the logs")
    # Params
    parser.add_argument("--num_episode", required=False,
                        default=DEFAULT_EPISODES, type=int,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 1)")
    parser.add_argument("--learning_rate", required=False,
                        default=DEFAULT_LR, type=float,
                        help="Learning rate for the Adam optimizer")

    return parser.parse_args()


def train(env,
          name=DEFAULT_NAME,
          iterations=DEFAULT_EPISODES,
          save_path=DEFAULT_SAVE_DIR,
          load_path=None,
          logs_path=DEFAULT_LOG_DIR,
          batch_size=DEFAULT_BATCH_SIZE,
          learning_rate=DEFAULT_LR,
          verbose=DEFAULT_VERBOSE):

    # Set config
    WGConfig.LR = learning_rate
    WGConfig.BATCH_SIZE = batch_size
    WGConfig.VERBOSE = verbose
    WGConfig.INITIAL_EPSILON = 1.0
    WGConfig.FINAL_EPSILON = 0.01
    WGConfig.DECAY_EPSILON = 1000
    WGConfig.UNIFORM_EPSILON = True
    WGConfig.K_RATIO = 256.0/134163.0
    WGConfig.UPDATE_FREQ = 16
    WGConfig.SIMULATE = -1
    WGConfig.SIMULATE_DO_NOTHING = False

    # Limit gpu usage
    limit_gpu_usage()

    agent = WGAgent(env.observation_space,
                    env.action_space,
                    name=name, 
                    is_training=True)

    if load_path is not None:
        agent.load(load_path)

    agent.train(env,
                iterations,
                save_path,
                logs_path)

if __name__ == "__main__":
    args = cli()

    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        print ("Fall back on default PandaPowerBackend")
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    env = make(args.data_dir,
               difficulty="competition",
               backend=backend,
               action_class=TopologyAndDispatchAction,
               reward_class=CombinedScaledReward,
               other_rewards = {
                   "game": GameplayReward,
                   "l2rpn": L2RPNReward,
                   "linereco": LinesReconnectedReward,
                   "overflow": CloseToOverflowReward
               })

    # Do not load entires scenario at once
    # (faster exploration)
    env.set_chunk_size(256)

    # Register custom reward for training
    cr = env.reward_helper.template_reward
    #cr.addReward("bridge", BridgeReward(), 1.0)
    #cr.addReward("distance", DistanceReward(), 1.0)
    #cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    cr.addReward("game", GameplayReward(), 2.0)
    #cr.addReward("eco", EconomicReward(), 2.0)
    cr.addReward("reco", LinesReconnectedReward(), 1.0)
    #cr.addReward("l2rpn", L2RPNReward(), 1.0 / env.n_line)
    cr.set_range(-1.0, 1.0)
    # Initialize custom rewards
    cr.initialize(env)

    train(env,
          name = args.name,
          iterations = args.num_episode,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate)
