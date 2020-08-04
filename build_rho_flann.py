#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.preprocessing import minmax_scale

import grid2op
from grid2op.Action import TopologyAndDispatchAction
from WolperGrid_Flann import WolperGrid_Flann
from wg_util import *


def cli():
    parser = argparse.ArgumentParser(description="Flann builder WolperGrid")

    parser.add_argument("--data_dir", required=False,
                        default="l2rpn_case14_sandbox",
                        help="Path to the dataset root directory")
    parser.add_argument("--action_file_in", required=True,
                        default="actions.npy",
                        help="Path to grid2op saved actions")
    parser.add_argument("--flann_index_out", required=True,
                        default="flann.index",
                        help="Path to flann index output")
    parser.add_argument("--flann_pts_out", required=True,
                        default="flann.npy",
                        help="Path to flann points output")
    parser.add_argument("--action_file_out", required=True,
                        default="actions.npy",
                        help="Path to grid2op actions output")

    return parser.parse_args()

def shuff(x):
    lx = len(x)
    s = np.random.choice(lx, size=lx, replace=False)
    return x[s]

if __name__ == "__main__":
    args = cli()

    # Pick fastest backend if possible
    try:
        from lightsim2grid.LightSimBackend import LightSimBackend
        backend = LightSimBackend()
    except:
        print ("Fall back on default PandaPowerBackend")
        from grid2op.Backend import PandaPowerBackend
        backend = PandaPowerBackend()

    # Create env
    env = grid2op.make(args.data_dir,
                       difficulty="competition",
                       backend=backend,
                       action_class=TopologyAndDispatchAction)

    # Shuffle training data
    env.chronics_handler.shuffle(shuffler=shuff)

    # Get observation
    obs = env.reset()

    # Create a flann tree instance
    wg_flann = WolperGrid_Flann(env.action_space, action_size=env.n_line)
    
    # Load actions
    actions_np = np.load(args.action_file_in)

    # Declare flann list
    act_flann = []
    act_ids = []

    # Process first DN action
    dn = env.action_space({})
    dn_sim ,_, done, _ = obs.simulate(dn)
    dn_rhos = dn_sim.rho
    act_flann.append(dn_rhos)
    act_ids.append(0)
    
    # Process rest of actions
    for act_id, action_np in enumerate(actions_np[1:]):
        # Build aciton from vect
        action = env.action_space.from_vect(action_np)
        # Simulate impact
        obs_sim, _, done, _ = obs.simulate(action)
        if done is False:
            # Get rhos
            action_rhos = np.array(obs_sim.rho)
            # Substract dn
            action_rhos -= dn_rhos
            # Save flann repr
            act_flann.append(action_rhos)
            act_ids.append(act_id)

    # Save actions that do not gameover
    np.save(args.action_file_out, actions_np[act_ids])
            
    # Make flann
    np_act_flann = np.array(act_flann)

    # Z-score standardization
    ## Compute mean/stddev
    mean = np.mean(np_act_flann, axis=0)
    std = np.std(np_act_flann, axis=0)
    ## Standardize 
    z_flann = (np_act_flann - mean) / std

    # MinMax normalization
    norm_flann = minmax_scale(np_act_flann,
                              feature_range=(0.0,1.0),
                              axis=0)

    # Register to flann index
    for act_flann in norm_flann:
        wg_flann.register_action(act_flann)

    # Build index
    wg_flann.construct_flann()

    # Save index
    wg_flann.save_flann(args.flann_index_out, args.flann_pts_out)
    
    

        
