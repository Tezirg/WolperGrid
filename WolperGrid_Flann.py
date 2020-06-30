#!/usr/bin/env python3

import math
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

        # Binary sizes
        self.bin_cut = 16
        self.bins_line = int(math.ceil(self.n_line / self.bin_cut))
        self.bins_bus = int(math.ceil(self.topo_size / self.bin_cut))
        self.bins_gens = int(math.ceil(self.disp_size / self.bin_cut))

        self.action_size_bin = 3 * self.bins_line
        self.action_size_bin += 3 * self.bins_bus
        self.action_size_bin += self.bins_gens + 1

        bin_max = np.ones(self.bin_cut, dtype=np.int32)
        self.bin_max = bin_max.dot(1 << np.arange(self.bin_cut)[::-1])
        self.bin_max = float(self.bin_max)

        # Declare variables
        self._act_vects = []
        self._flann = pf.FLANN()

        self.construct_vects()

    def _act_to_flann(self, act):
        # Declare zero vect
        act_v = np.zeros(self.action_size, dtype=np.float32)

        # Copy set line status
        line_s_f = act._set_line_status.astype(np.float32)
        act_v[self.line_s_offset[0]:self.line_s_offset[1]] = line_s_f

        # Copy change line status
        line_c_f = act._switch_line_status.astype(np.float32)
        act_v[self.line_c_offset[0]:self.line_c_offset[1]] = line_c_f
        act_v[self.line_c_offset[0]:self.line_c_offset[1]][line_c_f == 0.0] = -1.0
        # Copy set bus
        bus_s_f = act._set_topo_vect.astype(np.float32)
        act_v[self.bus_s_offset[0]:self.bus_s_offset[1]][bus_s_f == 1.0] = 1.0
        act_v[self.bus_s_offset[0]:self.bus_s_offset[1]][bus_s_f == 2.0] = -1.0

        # Copy change bus
        bus_c_f = act._change_bus_vect.astype(np.float32)
        act_v[self.bus_c_offset[0]:self.bus_c_offset[1]] = bus_c_f
        act_v[self.bus_c_offset[0]:self.bus_c_offset[1]][bus_c_f == 0.0] = -1.0

        # Dispatch linear rescale
        disp = disp_act_to_nn(self.action_space, act._redispatch)
        act_v[-self.disp_size:] = disp

        return act_v

    @staticmethod
    def _stdmtx(X):
        mean = X.mean()
        std = X.std(ddof=1)
        X = X - mean
        X = X / std
        return np.nan_to_num(X)

    @staticmethod
    def _normx(X):
        norm = np.linalg.norm(X)
        if norm < 1e-5:
            return X
        return X / norm

    def _vect_to_bins(self, n_bins, v):
        res = np.zeros(n_bins)
        for b in range(0, n_bins):
            bs = b * self.bin_cut
            be = (b + 1) * self.bin_cut
            cut = np.zeros(self.bin_cut)
            ce = self.bin_cut
            if be >= v.size:
                ce -= be - (v.size - 1)
                be = v.size - 1
            cut[0:ce] = v[bs:be]
            cut_bin = cut.dot(1 << np.arange(self.bin_cut)[::-1])
            res[b] = float(cut_bin) / self.bin_max
        return res

    def _act_to_flann_bin(self, act):
        # Declare zero vect
        act_v = np.zeros(self.action_size_bin, dtype=np.float32)

        off_s = 0
        off_e = self.bins_line
        line_s_on = act._set_line_status.astype(np.int32)
        line_s_on[line_s_on != 1] = 0
        bline_s_on = self._vect_to_bins(self.bins_line, line_s_on)
        act_v[off_s:off_e] = bline_s_on[:]

        off_s = off_e
        off_e += self.bins_line
        line_s_off = act._set_line_status.astype(np.int32) * -1
        line_s_off[line_s_off != 1] = 0
        bline_s_off = self._vect_to_bins(self.bins_line, line_s_off)
        act_v[off_s:off_e] = bline_s_off[:]

        off_s = off_e
        off_e += self.bins_line
        line_c = act._switch_line_status.astype(np.int32)
        bline_c = self._vect_to_bins(self.bins_line, line_c)
        act_v[off_s:off_e] = bline_c[:]

        off_s = off_e
        off_e += self.bins_bus
        bus_s_1 = act._set_topo_vect.astype(np.int32)
        bus_s_1[bus_s_1 != 1] = 0
        bbus_s_1 = self._vect_to_bins(self.bins_bus, bus_s_1)
        act_v[off_s:off_e] = bbus_s_1[:]

        off_s = off_e
        off_e += self.bins_bus
        bus_s_2 = act._set_topo_vect.astype(np.int32)
        bus_s_2[bus_s_2 != 2] = 0
        bbus_s_2 = self._vect_to_bins(self.bins_bus, bus_s_2)
        act_v[off_s:off_e] = bbus_s_2[:]

        off_s = off_e
        off_e += self.bins_bus
        bus_c = act._change_bus_vect.astype(np.int32)
        bbus_c = self._vect_to_bins(self.bins_bus, bus_c)
        act_v[off_s:off_e] = bbus_c[:]

        off_s = off_e
        off_e += self.bins_gens
        disp = disp_act_to_nn(self.action_space, act._redispatch)
        if np.any(disp != 0.0):
            disp_b = disp[disp != 0.0].astype(np.int32)
            act_v[off_s:off_e] = self._vect_to_bins(self.bins_gens, disp_b)
            act_v[-1] = disp[np.where(disp != 0.0)[0][0]]

        #print (act_v)
        return self._normx(act_v)

    def construct_vects(self):
        print("Flann build action vectors..")
        print("{} x {}".format(self.action_space.n, self.action_size_bin))
        
        for act in self.action_space.all_actions:
            act_v = self._act_to_flann_bin(act)
            # Add to list
            self._act_vects.append(act_v)

        self._flann_pts = np.array(self._act_vects)
        print("..Done")
        
    def construct_flann(self):
        print("Flann build tree..")
        pf.set_distance_type("euclidean")
        #pf.set_distance_type("manhattan")
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
        self._flann_pts = np.array(self._act_vects)
        self._flann.load_index(bytes_filename, self._flann_pts)

    def save_flann(self, filename):
        bytes_filename = filename.encode()
        self._flann.save_index(bytes_filename)

    def search_flann(self, act_vect, k):
        res, _ = self._flann.nn_index(act_vect, num_neighbors=k)
        # Dont care about distance
        return res        

    def __getitem__(self, index):
        return self._act_vects[index]
