#!/bin/sh

DATASET=$1

. src/scripts/local_env_setup.sh
python src/sgd.py --dataset "$DATASET"
