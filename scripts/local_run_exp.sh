#!/bin/sh

NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
PLOT=$3
NUM_WORKERS=$4

if test -z "$NUM_WORKERS"
then
      WORKERS=''
else
      WORKERS=' --num_workers '"$NUM_WORKERS"
fi

tmux new -s $NAME -d
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main master --master_socket_path /tmp/es_redis_master.sock --algo '$ALGO' --exp_file '"$EXP_FILE"' --plot '"$PLOT" C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m es_distributed.main workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo '"$ALGO"' '"$WORKERS" C-m
tmux a -t $NAME
