#!/bin/sh

NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
PLOT=$3

tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo '$ALGO' --exp_file '"$EXP_FILE"' --plot '"$PLOT" C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo '"$ALGO" C-m
tmux a -t $NAME
