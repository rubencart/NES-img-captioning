#!/bin/sh

# Launch an experiment on a local machine
# Arguments:
# ALGO:         nic_es | nic_nes
# EXP_FILE:     path from current directory to experiment json file with experiment settings
#               e.g. experiments/mscoco_nes.json
# NUM_WORKERS:  int number saying how many worker processes to use, will default to -1 (means 1 worker)
# ID:           chosen number to identify output files from this experiment, e.g. 11
# PORT:         redis port to use, can be 6379 | 6380 | 6381 | 6382
#               (to allow multiple experiments per machine being run concurrently)
#               will default to 6379

NAME=exp_`date "+%m_%d_%H_%M_%S"`
EXP_FILE=$1
NUM_WORKERS=$2
ID=$3
PORT=$4

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
tmux send-keys -t $NAME 'python -u src/main.py master --master_socket_path /tmp/es_redis_master_'"$PORT"'.sock --exp_file '"$EXP_FILE"' 2>&1 | tee ./output/'"$ID"'_master_outputfile.txt' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -u src/main.py workers --master_port '"$PORT"' --master_host localhost --relay_socket_path /tmp/es_redis_relay_'"$PORT"'.sock '"$WORKERS"' 2>&1 | tee ./output/'"$ID"'_worker_outputfile.txt' C-m
tmux a -t $NAME
