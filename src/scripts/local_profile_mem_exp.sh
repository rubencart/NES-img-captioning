#!/bin/sh

# Script to run an experiment with memory_profiler

NAME=exp_`date "+%m_%d_%H_%M_%S"`
EXP_FILE=$1
WORKERS=' --num_workers -1'
PORT=$2

cd src/

tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m memory_profiler main.py master --master_socket_path /tmp/es_redis_master_'"$PORT"'.sock --exp_file '"$EXP_FILE"' 2>&1 | tee ../output/master_p_outputfile.txt' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m memory_profiler main.py workers --master_port '"$PORT"' --master_host localhost --relay_socket_path /tmp/es_redis_relay_'"$PORT"'.sock '"$WORKERS"' 2>&1 | tee ../output/worker_p_outputfile.txt' C-m
tmux a -t $NAME
