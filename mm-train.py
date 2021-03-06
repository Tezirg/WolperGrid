#!/usr/bin/env python3

import argparse
import tensorflow as tf
import numpy as np

import grid2op
from grid2op.Reward import *
from grid2op.Action import *
from grid2op.Parameters import Parameters

from WolperGrid import WolperGrid as WGAgent
from WolperGrid_Config import WolperGrid_Config as WGConfig
from wg_util import limit_gpu_usage

DEFAULT_NAME = "WolperGrid"
DEFAULT_SAVE_DIR = "./models"
DEFAULT_LOG_DIR = "./logs-train"
DEFAULT_EPISODES = 10
DEFAULT_BATCH_SIZE = 128
DEFAULT_LR = 1e-4
DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Train baseline WolperGrid")

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
    parser.add_argument("--action_file", required=False,
                        default=None, type=str,
                        help="Path to pre-filtered action space")
    parser.add_argument("--flann_file", required=False,
                        default=None, type=str,
                        help="Path to pre-build flann index")

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
          action_path=None,
          flann_path=None,
          verbose=DEFAULT_VERBOSE):

    # Set config
    WGConfig.LR_CRITIC = 1e-4
    WGConfig.LR_ACTOR = 1e-5
    WGConfig.GRADIENT_CLIP = False
    WGConfig.BATCH_SIZE = batch_size
    WGConfig.VERBOSE = verbose
    WGConfig.INITIAL_EPSILON = 1.0
    WGConfig.FINAL_EPSILON = 0.001
    WGConfig.DECAY_EPSILON = 5000
    WGConfig.UNIFORM_EPSILON = True
    WGConfig.K = 56
    WGConfig.UPDATE_FREQ = 50
    WGConfig.ILLEGAL_GAME_OVER = False
    WGConfig.SIMULATE = -1
    WGConfig.SIMULATE_DO_NOTHING = False
    WGConfig.DISCOUNT_FACTOR = 0.97
    WGConfig.REPLAY_BUFFER_SIZE = 1024*128
    WGConfig.ACTION_SET = False
    WGConfig.ACTION_CHANGE = True
    WGConfig.ACTION_REDISP = True

    # Limit gpu usage
    limit_gpu_usage()

    agent = WGAgent(env.observation_space,
                    env.action_space,
                    action_file=action_path,
                    flann_file=flann_path,
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

    # Set custom params
    param = Parameters()
    #param.NO_OVERFLOW_DISCONNECTION = True

    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        print ("Fall back on default PandaPowerBackend")
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    env = grid2op.make(args.data_dir,
                       param=param,
                       backend=backend,
                       action_class=TopologyAndDispatchAction,
                       reward_class=CombinedScaledReward,
                       other_rewards = {
                           "game": GameplayReward,
                           "l2rpn": L2RPNReward,
                           "linereco": LinesReconnectedReward,
                           "overflow": CloseToOverflowReward
                       })

    for mix in env:
        # Do not load entires scenario at once
        # (faster exploration)
        mix.set_chunk_size(256)

        # Register custom reward for training
        cr = mix.reward_helper.template_reward
        #cr.addReward("bridge", BridgeReward(), 1.0)
        #cr.addReward("distance", DistanceReward(), 1.0)
        #cr.addReward("overflow", CloseToOverflowReward(), 1.0)
        gp = GameplayReward()
        gp.set_range(-4.0, 1.5)
        cr.addReward("game", gp, 1.0)
        #cr.addReward("eco", EconomicReward(), 2.0)
        reco = LinesReconnectedReward()
        reco.set_range(-1.0, 2.5)
        cr.addReward("reco", reco, 2.0)
        l2 = L2RPNReward()
        l2.set_range(0.0, env.n_line)
        cr.addReward("l2rpn", l2, 1.0 / env.n_line)
        #cr.addReward("flat", IncreasingFlatReward(), 1.0 / 8063.0)
        cr.set_range(-1.0, 1.0)
        # Initialize custom rewards
        cr.initialize(mix)

        # Shuffle training data
        def shuff(x):
            lx = len(x)
            s = np.random.choice(lx, size=lx, replace=False)
            return x[s]
        mix.chronics_handler.shuffle(shuffler=shuff)

    train(env,
          name = args.name,
          iterations = args.num_episode,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate,
          action_path = args.action_file,
          flann_path = args.flann_file)
