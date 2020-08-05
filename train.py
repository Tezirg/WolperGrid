#!/usr/bin/env python3

import argparse
import tensorflow as tf
import numpy as np

from grid2op.MakeEnv import make
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
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR = 1e-5
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
    parser.add_argument("--flann_index_file", required=False,
                        default=None, type=str,
                        help="Path to pre-build flann index")
    parser.add_argument("--flann_pts_file", required=False,
                        default=None, type=str,
                        help="Path to pre-build flann points")
    parser.add_argument("--flann_action_size", required=False,
                        default=None, type=int,
                        help="Pre-build flann representation size")
    # Params
    parser.add_argument("--num_episode", required=False,
                        default=DEFAULT_EPISODES, type=int,
                        help="Number of training iterations")
    parser.add_argument("--batch_size", required=False,
                        default=DEFAULT_BATCH_SIZE, type=int,
                        help="Mini batch size (defaults to 32)")
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
          flann_index_path=None,
          flann_pts_path=None,
          flann_action_size=None,
          verbose=DEFAULT_VERBOSE):

    # Set config
    WGConfig.LR_CRITIC = 1e-5
    WGConfig.LR_ACTOR = 1e-5
    WGConfig.GRADIENT_CLIP = False
    WGConfig.GRADIENT_INVERT = True
    WGConfig.BATCH_SIZE = 64
    WGConfig.VERBOSE = verbose
    WGConfig.INITIAL_EPSILON = 1.0
    WGConfig.FINAL_EPSILON = 0.02
    WGConfig.DECAY_EPSILON = 500
    WGConfig.UNIFORM_EPSILON = True
    WGConfig.K = 32
    WGConfig.UPDATE_FREQ = 16
    WGConfig.LOG_FREQ = WGConfig.UPDATE_FREQ * 10
    WGConfig.UPDATE_TARGET_SOFT_TAU = 1e-3
    WGConfig.ILLEGAL_GAME_OVER = False
    WGConfig.SIMULATE = -1
    WGConfig.SIMULATE_DO_NOTHING = False
    WGConfig.DISCOUNT_FACTOR = 0.99
    WGConfig.REPLAY_BUFFER_SIZE = 1024*128
    WGConfig.REPLAY_BUFFER_MIN = 1024*3
    WGConfig.ACTION_SET_LINE = False
    WGConfig.ACTION_CHANGE_LINE = False
    WGConfig.ACTION_SET_BUS = True
    WGConfig.ACTION_CHANGE_BUS = False
    WGConfig.ACTION_REDISP = False

    agent = WGAgent(env.observation_space,
                    env.action_space,
                    action_file=action_path,
                    flann_index_file=flann_index_path,
                    flann_pts_file=flann_pts_path,
                    flann_action_size=flann_action_size,
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

    # Limit gpu usage
    limit_gpu_usage()

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
    env.set_chunk_size(128)

    # Register custom reward for training
    cr = env.reward_helper.template_reward
    #cr.addReward("bridge", BridgeReward(), 1.0)
    #cr.addReward("distance", DistanceReward(), 1.0)
    #cr.addReward("overflow", CloseToOverflowReward(), 1.0)
    gp = GameplayReward()
    gp.set_range(-5.0, 1.5)
    cr.addReward("game", gp, 1.0)
    #cr.addReward("eco", EconomicReward(), 2.0)
    #reco = LinesReconnectedReward()
    #reco.set_range(-1.0, 1.0)
    #cr.addReward("reco", reco, 0.25)
    l2 = L2RPNReward()
    l2.set_range(0.0, env.n_line)
    cr.addReward("l2rpn", l2, 1.0 / env.n_line)
    #cr.addReward("flat", IncreasingFlatReward(), 1.0 / 8063.0)
    cr.set_range(-10.0, 5.0)
    # Initialize custom rewards
    cr.initialize(env)

    # Shuffle training data
    def shuff(x):
        lx = len(x)
        s = np.random.choice(lx, size=lx, replace=False)
        return x[s]
    env.chronics_handler.shuffle(shuffler=shuff)

    train(env,
          name = args.name,
          iterations = args.num_episode,
          save_path = args.save_dir,
          load_path = args.load_file,
          logs_path = args.logs_dir,
          batch_size = args.batch_size,
          learning_rate = args.learning_rate,
          action_path = args.action_file,
          flann_index_path = args.flann_index_file,
          flann_pts_path = args.flann_pts_file,
          flann_action_size = args.flann_action_size)
