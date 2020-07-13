#!/usr/bin/env python3

import os
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from grid2op.Episode import EpisodeData

DEFAULT_VERBOSE = True

def cli():
    parser = argparse.ArgumentParser(description="Action impact distribution")
    parser.add_argument("--agent_path", required=True,
                        help="Path to runner output logs directory") 
    parser.add_argument("--episode_name", required=True,
                        help="Name of the episode logs directory") 
    parser.add_argument("--verbose", action='store_true',
                        help="Verbose runner output")
    return parser.parse_args()

def distrib(agent_path,
            episode_name,
            verbose=DEFAULT_VERBOSE):
    ep = EpisodeData.from_disk(agent_path, episode_name)

    tree = [
        [ ],
        [ ],
        [ ]
    ]
    for a in ep.actions:
        c = -1
        # Find category & index
        ## Lines set
        if np.any(a._set_line_status != 0):
            c = 0
            i = np.where(a._set_line_status != 0)[0][0]
        ## Lines change
        elif np.any(a._switch_line_status):
            c = 0
            i = np.where(a._switch_line_status == True)[0][0]
        ## Topo set
        elif np.any(a._set_topo_vect != 0):
            c = 1
            obj = np.where(a._set_topo_vect != 0)[0][0]
            _, _, i = a._obj_caract_from_topo_id(obj)
        ## Topo change
        elif np.any(a._change_bus_vect):
            c = 1
            obj = np.where(a._change_bus_vect == True)[0][0]
            _, _, i = a._obj_caract_from_topo_id(obj)
        ## Gen redispatch
        elif np.any(a._redispatch != 0.0):
            c = 2
            i = np.where(a._redispatch != 0.0)[0][0]

        # Register it into tree
        if c != -1:
            tree[c].append(i)

    plt.figure(1)    
    fig_lines, ax_lines = plt.subplots()
    ax_lines.hist(tree[0],
                  range=(0, ep.observation_space.n_line),
                  align="left",
                  bins=ep.observation_space.n_line,
                  density=True)    
    ax_lines.set_xlabel('Lines')
    ax_lines.set_xticks(np.arange(ep.observation_space.n_line))
    ax_lines.set_ylabel('Actions')
    
    plt.figure(2)
    fig_subs, ax_subs = plt.subplots()
    ax_subs.hist(tree[1],
                 range=(0, ep.observation_space.n_sub),
                 bins=ep.observation_space.n_sub,
                 align="left",
                 density=True)
    ax_subs.set_xlabel('Substations')
    ax_subs.set_xticks(np.arange(ep.observation_space.n_sub))
    ax_subs.set_ylabel('Actions')


    plt.figure(3)
    fig_gens, ax_gens = plt.subplots()
    ax_gens.hist(tree[2],
                 range=(0, ep.observation_space.n_gen),
                 align="left",
                 bins=ep.observation_space.n_gen,
                 density=True)
    ax_gens.set_xlabel('Generators')
    ax_gens.set_xticks(np.arange(ep.observation_space.n_gen))
    ax_gens.set_ylabel('Actions')

    plt.show()

if __name__ == "__main__":
    # Parse command line
    args = cli()

    distrib(args.agent_path,
            args.episode_name,
            args.verbose)
