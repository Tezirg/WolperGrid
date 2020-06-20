#!/usr/bin/env python3

import os
import argparse
import tensorflow as tf

from grid2op.MakeEnv import make
from grid2op.Runner import Runner
from grid2op.Reward import *
from grid2op.Action import *

from WolperGrid import WolperGrid as WGAgent
from WolperGrid_Config import WolperGrid_Config as WGAgentConf
from l2rpn_baselines.utils.save_log_gif import save_log_gif
from wg_util import limit_gpu_usage

DEFAULT_LOGS_DIR = "./logs-eval"
DEFAULT_NB_EPISODE = 1
DEFAULT_NB_PROCESS = 1
DEFAULT_MAX_STEPS = -1
DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Eval baseline GridRDQN")
    parser.add_argument("--data_dir", required=True,
                        help="Path to the dataset root directory")
    parser.add_argument("--load_dir", required=True,
                        help="The path to the actor/critic models")
    parser.add_argument("--logs_dir", required=False,
                        default=DEFAULT_LOGS_DIR, type=str,
                        help="Path to output logs directory") 
    parser.add_argument("--nb_episode", required=False,
                        default=DEFAULT_NB_EPISODE, type=int,
                        help="Number of episodes to evaluate")
    parser.add_argument("--nb_process", required=False,
                        default=DEFAULT_NB_PROCESS, type=int,
                        help="Number of cores to use")
    parser.add_argument("--max_steps", required=False,
                        default=DEFAULT_MAX_STEPS, type=int,
                        help="Maximum number of steps per scenario")
    parser.add_argument("--gif", action='store_true',
                        help="Enable GIF Output")
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    return parser.parse_args()

def evaluate(env,
             load_path=None,
             logs_path=DEFAULT_LOGS_DIR,
             nb_episode=DEFAULT_NB_EPISODE,
             nb_process=DEFAULT_NB_PROCESS,
             max_steps=DEFAULT_MAX_STEPS,
             verbose=DEFAULT_VERBOSE,
             save_gif=False):

    # Limit gpu usage
    limit_gpu_usage()

    WGAgentConf.VERBOSE = verbose
    WGAgentConf.K_RATIO = 256.0/134163.0
    WGAgentConf.SIMULATE = -1
    WGAgentConf.SIMULATE_DO_NOTHING = False

    runner_params = env.get_params_for_runner()
    runner_params["verbose"] = verbose

    # Run
    # Create agent
    agent = WGAgent(env.observation_space,
                    env.action_space,
                    is_training=False)

    # Load weights from file
    agent.load(load_path)

    # Build runner
    runner = Runner(**runner_params,
                    agentClass=None,
                    agentInstance=agent)

    # Print model summary
    if verbose:
        stringlist = []
        agent.Qmain.actor.summary(print_fn=lambda x: stringlist.append(x))
        stringlist.append(" -- ")
        agent.Qmain.critic.summary(print_fn=lambda x: stringlist.append(x))
        short_model_summary = "\n".join(stringlist)
        print(short_model_summary)

    # Run
    os.makedirs(logs_path, exist_ok=True)
    res = runner.run(path_save=logs_path,
                     nb_episode=nb_episode,
                     nb_process=nb_process,
                     max_iter=max_steps,
                     pbar=verbose)

    # Print summary
    if verbose:
        print("Evaluation summary:")
        for _, chron_name, cum_reward, nb_tstep, max_ts in res:
            msg_tmp = "chronics at: {}".format(chron_name)
            msg_tmp += "\ttotal reward: {:.6f}".format(cum_reward)
            msg_tmp += "\ttime steps: {:.0f}/{:.0f}".format(nb_tstep, max_ts)
            print(msg_tmp)

    if save_gif:
        save_log_gif(logs_path, res)

    return res

if __name__ == "__main__":
    # Parse command line
    args = cli()

    # Try to use faster simulator
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        print ("Fall back on default PandaPowerBackend")
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    # Create dataset env
    env = make(args.data_dir,
               backend=backend,
               reward_class=L2RPNSandBoxScore,
               action_class=TopologyAction,
               other_rewards={
                   "game": GameplayReward,
                   "l2rpn": L2RPNReward,
                   "overflow": CloseToOverflowReward,
                   "wcci_score": L2RPNSandBoxScore
               })
    # Call evaluation interface
    evaluate(env,
             load_path=args.load_dir,
             logs_path=args.logs_dir,
             nb_episode=args.nb_episode,
             nb_process=args.nb_process,
             max_steps=args.max_steps,
             verbose=args.verbose,
             save_gif=args.gif)
