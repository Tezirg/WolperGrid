#!/bin/bash

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

export WG_NAME=wg-X.0.0.x
export WG_DATA=~/data_grid2op/rte_case14_realistic

./inspect_action_space.py --path_data $WG_DATA

rm -rf ./logs-train/$WG_NAME
./train.py\
    --name $WG_NAME \
    --data_dir $WG_DATA \
    --num_pre_steps 256 \
    --num_train_steps 131072 \
    --num_frames 4

rm -rf ./logs-eval/$WG_NAME
./evaluate.py \
    --data_dir $WG_DATA \
    --load_file ./models/$WG_NAME.tf \
    --logs_dir ./logs-eval/$WG_NAME \
    --nb_episode 10
