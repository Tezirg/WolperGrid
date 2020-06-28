#!/usr/bin/env python3

import numpy as np
import pyflann as pf
import grid2op

from wg_util import *

class WolperGrid_Flann(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.n_line = action_space.n_line
        self.topo_size = action_space.dim_topo
        self.disp_size = action_space.n_gen
        
        # Compute sizes and offsets once
        self.action_size = self.n_line * 2 + \
            self.topo_size * 2 + \
            self.disp_size

        self.line_s_offset = [
            0,
            self.n_line
        ]
        self.line_c_offset = [
            self.line_s_offset[1],
            self.line_s_offset[1] + self.n_line
        ]
        self.bus_s_offset = [
            self.line_c_offset[1],
            self.line_c_offset[1] + self.topo_size
        ]
        self.bus_c_offset = [
            self.bus_s_offset[1],
            self.bus_s_offset[1] + self.topo_size
        ]

        # Declare variables
        self._act_vects = []
        self._flann = pf.FLANN()

        self.construct_vects()
        # TODO: Do not build if loaded from disk
        self.construct_flann()

    def _act_to_flann(self, act):
        # Declare zero vect
        act_v = np.zeros(self.action_size, dtype=np.float32)
            
        # Copy set line status
        line_s_f = act._set_line_status.astype(np.float32)
        act_v[self.line_s_offset[0]:self.line_s_offset[1]] = line_s_f
        
        # Copy change line status
        line_c_f = act._switch_line_status.astype(np.float32)
        act_v[self.line_c_offset[0]:self.line_c_offset[1]] = line_c_f
        
        # Copy set bus
        bus_s_f = act._set_topo_vect.astype(np.float32)
        act_v[self.bus_s_offset[0]:self.bus_s_offset[1]] = bus_s_f
        act_v[self.bus_s_offset[0]:self.bus_s_offset[1]][bus_s_f == 2.0] = -1.0
        
        # Copy change bus
        bus_c_f = act._change_bus_vect.astype(np.float32)
        act_v[self.bus_c_offset[0]:self.bus_c_offset[1]] = bus_c_f
        
        # Dispatch linear rescale
        disp = disp_act_to_nn(self.action_space, act._redispatch)
        act_v[-self.disp_size:] = disp

        return act_v

    def construct_vects(self):
        print("Flann build action vectors..")
        
        for act in self.action_space.all_actions:
            act_v = self._act_to_flann(act)
            # Add to list
            self._act_vects.append(act_v)
            
        print("act_size {}".format(self.action_size))
        print("..Done")
        
    def construct_flann(self):
        print("Flann build tree..")
        flann_pts = np.array(self._act_vects)
        pf.set_distance_type("manhattan")
        self._flann.build_index(flann_pts,
                                algorithm="kmeans",
                                iterations=7,
                                cb_index=0.5,
                                centers_init="kmeanspp",
                                branching=32,
                                checks=16)
        print("..Done")

    def load_flann(self, filename):
        flann_pts = np.array(self._act_vects)
        self._flann.load_index(filename)

    def save_flann(self, filename):
        self._flann.save_index(filename)

    def search_flann(self, act_vect, k):
        res, _ = self._flann.nn_index(act_vect, num_neighbors=k)
        # Dont care about distance
        return res        

    def __getitem__(self, index):
        return self._act_vects[index]
