#!/usr/bin/env python3

import argparse
import numpy as np
from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
import pandas as pd

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
                       difficulty="0",
                       backend=backend,
                       action_class=TopologyAndDispatchAction)
    env.seed(42)
    np.random.seed(42)

    # Shuffle training data
    env.chronics_handler.shuffle(shuffler=shuff)
    
    # Load actions
    actions_np = np.load(args.action_file_in)

    # Declare store
    act_rhos = {}
    for act_id in range(actions_np.shape[0]):
        act_rhos[act_id] = []

    # Process actions on N scenarios
    n_scenario = 10
    for _ in range(n_scenario):
        # Get observation, new scenario
        obs = env.reset()

        # Process first DN action
        dn = env.action_space({})
        dn_sim ,_, done, _ = obs.simulate(dn)

        # Skip if dn gameover, no reference point
        if done:
            continue

        # Register dn rhos
        dn_rhos = dn_sim.rho
        act_rhos[0].append(np.zeros(env.n_line))
    
        # Process rest of actions
        for act_id, action_np in enumerate(actions_np[1:]):
            # Build aciton from vect
            action = env.action_space.from_vect(action_np)
            # Simulate impact
            obs_sim, _, done, _ = obs.simulate(action)
            if done:
                continue
            # Get rhos
            action_rhos = np.array(obs_sim.rho)
            # Substract dn
            action_rhos -= dn_rhos
            # Save flann repr
            act_rhos[act_id].append(action_rhos)


    # Declare FLANN stores
    act_flann = []
    act_ids = []

    # Mean rhos
    for act_id, rhos_li in act_rhos.items():
        if len(rhos_li) == 0:
            continue
        act_rhos_mean = np.mean(np.array(rhos_li), axis=0)
        act_ids.append(act_id)
        act_flann.append(act_rhos_mean)

    # Save actions that do not gameover all the time
    actions_flann = actions_np[act_ids]
    np.save(args.action_file_out, actions_flann)
    print("Saved {} actions".format(len(act_flann)))

    # Make flann
    np_act_flann = np.array(act_flann)

    # Z-score standardization
    ## Compute mean/stddev
    mean = np.mean(np_act_flann, axis=0)
    std = np.std(np_act_flann, axis=0)
    ## Standardize 
    z_flann = (np_act_flann - mean) / std
    print("Z shape = ", z_flann.shape)
    
    # MinMax normalization [0;1]
    norm_flann = minmax_scale(np_act_flann,
                              feature_range=(-1.0,1.0),
                              axis=0)
    print("MinMax shape = ", norm_flann.shape)

    # PCA reduction
    pca_size = 16
    pca = PCA(n_components=pca_size)
    pca_flann = pca.fit_transform(np_act_flann)
    print ("PCA shape = ", pca_flann.shape)
    pca_norm_flann = minmax_scale(pca_flann,
                                  feature_range=(-1.0,1.0),
                                  axis=0)

    # Dump to csv for debug
    with pd.ExcelWriter('flann_repr.xlsx') as writer:
        df1 = pd.DataFrame(z_flann,
                           index=act_ids,
                           columns=env.name_line)
        df1.to_excel(writer, sheet_name="Z-score std")
        df2 = pd.DataFrame(norm_flann,
                           index=act_ids,
                           columns=env.name_line)
        df2.to_excel(writer, sheet_name="MinMaxScale")
        df3 = pd.DataFrame(pca_flann,
                           index=act_ids,
                           columns=np.arange(pca_size))
        df3.to_excel(writer, sheet_name="PCA {}".format(pca_size))
        df4 = pd.DataFrame(pca_norm_flann,
                           index=act_ids,
                           columns=np.arange(pca_size))
        df4.to_excel(writer, sheet_name="PCA {} MinMaxScale".format(pca_size))

    # Create a flann tree instance
    wg_flann = WolperGrid_Flann(env.action_space, action_size=pca_size)

    # Register to flann index
    for register_flann in pca_norm_flann:
        wg_flann.register_action(register_flann)

    # Build index
    wg_flann.construct_flann()

    # Save index
    wg_flann.save_flann(args.flann_index_out, args.flann_pts_out)
