#!/usr/bin/env python3

import math
import numpy as np
import pyflann as pf
import grid2op

from WolperGrid_Config import WolperGrid_Config as cfg
from wg_util import *

class WolperGrid_Flann(object):
    def __init__(self, action_space, action_size=None):
        self.action_space = action_space
        self.max_elems = np.amax(self.action_space.sub_info)
        self.n_line = action_space.n_line
        self.topo_size = action_space.dim_topo
        self.disp_size = action_space.n_gen

        # Declare variables
        self._act_flann = []
        self._flann_pts = None
        self._flann = pf.FLANN()

        if action_size is None:
            # Compute sizes and offsets once
            self.act_offset = [0]
            self._action_size = 0

            if cfg.ACTION_SET_LINE:
                self._action_size += self.n_line
                self.act_offset.append(self.act_offset[-1])
                self.act_offset.append(self.act_offset[-1] + self.n_line)

            if cfg.ACTION_CHANGE_LINE:
                self._action_size += self.n_line
                self.act_offset.append(self.act_offset[-1])
                self.act_offset.append(self.act_offset[-1] + self.n_line)

            if cfg.ACTION_SET_BUS:
                self._action_size += self.topo_size
                self.act_offset.append(self.act_offset[-1])
                self.act_offset.append(self.act_offset[-1] + self.topo_size)

            if cfg.ACTION_CHANGE_BUS:
                self._action_size += self.topo_size
                self.act_offset.append(self.act_offset[-1])
                self.act_offset.append(self.act_offset[-1] + self.topo_size)

            if cfg.ACTION_REDISP:
                self._action_size += self.disp_size
                self.act_offset.append(self.act_offset[-1])
                self.act_offset.append(self.act_offset[-1] + self.disp_size)

            # Expose proto size
            self.action_size = self._action_size
        
            self.construct_vects()
        else:
            self.action_size = action_size
            self._action_size = action_size

    def _act_to_flann(self, act):
        # Declare zero vect
        act_v = np.full(self._action_size, 0.0, dtype=np.float32)

        off_s = 1
        off_e = 2

        if cfg.ACTION_SET_LINE: # Copy set line status
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            line_s_f = act._set_line_status.astype(int)
            act_v[act_s:act_e][line_s_f == -1] = -1.0
            act_v[act_s:act_e][line_s_f == 0] = 0.0
            act_v[act_s:act_e][line_s_f == 1] = -1.0
            off_s += 2
            off_e += 2

        if cfg.ACTION_CHANGE_LINE: # Copy change line status
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            act_v[act_s:act_e] = act._switch_line_status.astype(np.float32)
            off_s += 2
            off_e += 2

        if cfg.ACTION_SET_BUS: # Copy set bus
            act_s = self.act_offset[off_s]
            act_e = self.act_offset[off_e] 
            bus_s_f = act._set_topo_vect.astype(int)
            act_v[act_s:act_e][bus_s_f == 0] = -1.0
            act_v[act_s:act_e][bus_s_f == 1] = 0.0
            act_v[act_s:act_e][bus_s_f == 2] = 1.0
            off_s += 2
            off_e += 2

        if cfg.ACTION_CHANGE_BUS: # Copy change bus
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
        if norm == 0.0:
            return X
        return X / norm

    def register_action(self, act_flann):
        self._act_flann.append(act_flann)
    
    def construct_vects(self):
        print("Flann build action vectors..")
        print("{} x {}".format(self.action_space.n, self.action_size))
        
        for act in self.action_space.all_actions:
            act_flann = self._act_to_flann(act)
            self.register_action(act_flann)
            
        print("..Done")
        
    def construct_flann(self):
        print("Flann build tree..")
        pf.set_distance_type("euclidean")
        self._flann_pts = np.array(self._act_flann)
        self._flann.build_index(self._flann_pts,
                                algorithm="kmeans",
                                iterations=11,
                                cb_index=0.5,
                                centers_init="kmeanspp",
                                branching=32,
                                checks=16)
        print("..Done")

    def load_flann(self, index_filename, points_filename):
        self._flann_pts = np.load(points_filename)
        bytes_index_filename = index_filename.encode()
        self._flann.load_index(bytes_index_filename, self._flann_pts)
        self._act_flann = list(self._flann_pts)

    def save_flann(self, index_filename, points_filename):
        np.save(points_filename, self._flann_pts)
        bytes_index_filename = index_filename.encode()
        self._flann.save_index(bytes_index_filename)

    def search_flann(self, act_vect, k):
        search_vect = np.array(act_vect)
        res, _ = self._flann.nn_index(search_vect, num_neighbors=k)
        # Dont care about distance
        return res        

    def __getitem__(self, index):
        return self._act_flann[index]
