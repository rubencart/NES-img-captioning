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
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m    # -m src.main master
tmux send-keys -t $NAME 'python src/main.py master --master_socket_path /tmp/es_redis_master.sock --algo '$ALGO' --exp_file '"$EXP_FILE" C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python src/main.py workers --master_host localhost --relay_socket_path /tmp/es_redis_relay.sock --algo '"$ALGO"' '"$WORKERS" C-m
tmux a -t $NAME
