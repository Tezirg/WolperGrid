#!/bin/bash

# Copyright (c) 2020, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# This Source Code Form is subject to the terms of the Mozilla Public License, version 2.0.
# If a copy of the Mozilla Public License, version 2.0 was not distributed with this file,
# you can obtain one at http://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
# This file is part of L2RPN Baselines, L2RPN Baselines a repository to host baselines for l2rpn competitions.

export WG_NAME=wg-X.0.0.x
export WG_DATA=l2rpn_neurips_2020_track2_large

./inspect_action_space.py --path_data $WG_DATA

rm -rf ./logs-train/$WG_NAME
# First time
./mm-train.py\
    --name $WG_NAME \
    --data_dir $WG_DATA \
    --num_episode 10000

# Flann update
./mm-train.py\
    --name $WG_NAME \
    --data_dir $WG_DATA \
    --action_file ./models/$WG_NAME/actions.npy \
    --num_episode 100000

# Hypers
./mm-train.py\
    --name $WG_NAME \
    --data_dir $WG_DATA \
    --action_file ./models/$WG_NAME/actions.npy \
    --flann_file ./models/$WG_NAME/flann.index \
    --num_episode 100000

rm -rf ./logs-eval/$WG_NAME
./mm-evaluate.py \
    --verbose \
    --data_dir $WG_DATA \
    --load_dir ./models/$WG_NAME/ \
    --load_action ./models/$WG_NAME/actions.npy \
    --load_flann ./models/$WG_NAME/flann.index \
    --logs_dir ./logs-eval/$WG_NAME \
    --nb_episode 12
