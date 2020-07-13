#!/usr/bin/env python3

import math
import numpy as np
import pyflann as pf
import grid2op

from WolperGrid_Config import WolperGrid_Config as cfg
from wg_util import *

class WolperGrid_Flann(object):
    def __init__(self, action_space):
        self.action_space = action_space
        self.n_line = action_space.n_line
        self.topo_size = action_space.dim_topo
        self.disp_size = action_space.n_gen

        # Compute sizes and offsets once
        flann_diff = []
        self.act_offset = [0]
        self._action_size = 0

        if cfg.ACTION_SET:
            self._action_size += self.n_line
            self.act_offset.append(self.act_offset[-1])
            self.act_offset.append(self.act_offset[-1] + self.n_line)
            flann_diff.append(np.full(self.n_line, 10.0, dtype=np.float32))

        if cfg.ACTION_CHANGE:
            self._action_size += self.n_line
            self.act_offset.append(self.act_offset[-1])
            self.act_offset.append(self.act_offset[-1] + self.n_line)
            flann_diff.append(np.full(self.n_line, 10.0, dtype=np.float32))

        if cfg.ACTION_SET:
            self._action_size += self.topo_size * 2
            self.act_offset.append(self.act_offset[-1])
            self.act_offset.append(self.act_offset[-1] + self.topo_size * 2)
            bus_idx = np.arange(self.topo_size * 2)
            self.s_bus0 = bus_idx[bus_idx % 2 == 0]
            self.s_bus1 = bus_idx[bus_idx % 2 == 1]            
            flann_diff.append(np.full(self.topo_size, 100.0, dtype=np.float32))

        if cfg.ACTION_CHANGE:
            self._action_size += self.topo_size
            self.act_offset.append(self.act_offset[-1])
            self.act_offset.append(self.act_offset[-1] + self.topo_size)
            flann_diff.append(np.full(self.topo_size, 100.0, dtype=np.float32))

        if cfg.ACTION_REDISP:
            self._action_size += self.disp_size
            self.act_offset.append(self.act_offset[-1])
            self.act_offset.append(self.act_offset[-1] + self.disp_size)
            flann_diff.append(np.full(self.disp_size, 100.0, dtype=np.float32))
        
        self.flann_diff = np.concatenate(flann_diff)

        # Expose proto size
        self.action_size = self._action_size

        # Declare variables
        self._act_vects = []
        self._flann = pf.FLANN()

        self.construct_vects()

    def _act_to_flann(self, act):
        # Declare zero vect
        act_v = np.zeros(self._action_size, dtype=np.float32)

        off_s = 1
        off_e = 2

        if cfg.ACTION_SET: # Copy set line status
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            line_s_f = act._set_line_status.astype(np.float32)
            act_v[act_s:act_e] = line_s_f
            off_s += 2
            off_e += 2

        if cfg.ACTION_CHANGE: # Copy change line status
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            act_v[act_s:act_e] = act._switch_line_status.astype(np.float32)
            off_s += 2
            off_e += 2

        if cfg.ACTION_SET: # Copy set bus
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            bus_s_f = act._set_topo_vect.astype(int)
            act_v[act_s:act_e][self.s_bus0][bus_s_f == 1] = 1.0
            act_v[act_s:act_e][self.s_bus1][bus_s_f == 2] = 1.0
            off_s += 2
            off_e += 2

        if cfg.ACTION_CHANGE: # Copy change bus
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e]
            act_v[act_s:act_e] = act._change_bus_vect.astype(np.float32)
            off_s += 2
            off_e += 2

        if cfg.ACTION_REDISP: # Dispatch linear rescale
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            disp = disp_act_to_nn(self.action_space, act._redispatch)
            act_v[act_s:act_e] = disp

        return act_v

    @staticmethod
    def _normx(X):
        norm = np.linalg.norm(X)
        if norm < 1.0:
            return X
        return X / norm

    def construct_vects(self):
        print("Flann build action vectors..")
        print("{} x {}".format(self.action_space.n, self.action_size))
        
        for act in self.action_space.all_actions:
            act_v = self._act_to_flann(act)
            # Add to list
            self._act_vects.append(act_v)

        flann_pts = [np.array(a) for a in self._act_vects]
        self._flann_pts = np.array(flann_pts)# * self.flann_diff
        print("{} ..Done".format(self._flann_pts.shape))
        
    def construct_flann(self):
        print("Flann build tree..")
        pf.set_distance_type("euclidean")
        self._flann.build_index(self._flann_pts,
                                algorithm="kmeans",
                                iterations=11,
                                cb_index=0.5,
                                centers_init="kmeanspp",
                                branching=32,
                                checks=16)
        print("..Done")

    def load_flann(self, filename):
        bytes_filename = filename.encode()
        self._flann.load_index(bytes_filename, self._flann_pts)

    def save_flann(self, filename):
        bytes_filename = filename.encode()
        self._flann.save_index(bytes_filename)

    def search_flann(self, act_vect, k):
        search_vect = np.array(act_vect)# * self.flann_diff
        res, _ = self._flann.nn_index(search_vect, num_neighbors=k)
        # Dont care about distance
        return res        

    def __getitem__(self, index):
        return self._act_vects[index]
