#!/bin/sh

./build_rho_flann.py --data_dir ~/data_grid2op/l2rpn_neurips_2020_track2_large/x1.0 --action_file_in ./models/wg-cv-0014/actions.npy --flann_index_out ./dev.index --flan_pts_out ./flann.npy --action_file_out ./dev.npy
