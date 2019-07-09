#!/bin/sh

NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
NUM_WORKERS=$3
ID=$4
PORT=$5

if test -z "$ID"
then
      ID=0
fi

if test -z "$PORT"
then
      PORT=6379
fi

if test -z "$NUM_WORKERS"
then
      WORKERS=''
else
      WORKERS=' --num_workers '"$NUM_WORKERS"
fi

tmux new -s $NAME -d
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -u src/main.py master --master_socket_path /tmp/es_redis_master_'"$PORT"'.sock --algo '$ALGO' --exp_file '"$EXP_FILE"' 2>&1 | tee ./output/'"$ID"'_master_outputfile.txt' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -u src/main.py workers --master_port '"$PORT"' --master_host localhost --relay_socket_path /tmp/es_redis_relay_'"$PORT"'.sock --algo '"$ALGO"' '"$WORKERS"' 2>&1 | tee ./output/'"$ID"'_worker_outputfile.txt' C-m
tmux a -t $NAME
