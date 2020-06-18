#!/bin/bash

export WG_NAME=wg-X.0.0.x
export WG_DATA=~/data_grid2op/l2rpn_neurips_2020_track2

tree -L 2 $WG_DATA

rm -rf ./logs-train/$WG_NAME
./train.py\
    --name $WG_NAME \
    --data_dir $WG_DATA \
    --num_episode 1000

rm -rf ./logs-eval/$WG_NAME
./evaluate.py \
    --verbose \
    --data_dir $WG_DATA \
    --load_dir ./models/$WG_NAME/ \
    --logs_dir ./logs-eval/$WG_NAME \
    --nb_episode 10
