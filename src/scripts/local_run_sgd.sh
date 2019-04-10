#!/bin/sh

DATASET=$1
EPOCHS=$2

. src/scripts/local_env_setup.sh
python src/sgd.py --dataset "$DATASET" --epochs "$EPOCHS" 2>&1 | tee ./output/sgd_outputfile.txt
