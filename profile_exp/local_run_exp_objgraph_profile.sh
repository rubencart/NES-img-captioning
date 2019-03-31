#!/bin/sh

NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
NUM_WORKERS=$3

if test -z "$NUM_WORKERS"
then
      WORKERS=''
else
      WORKERS=' --num_workers '"$NUM_WORKERS"
fi

tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m cProfile -o tmp.txt profile_exp/profile_master.py' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m cProfile -o tmp.txt profile_exp/profile_worker.py' C-m
tmux a -t $NAME
