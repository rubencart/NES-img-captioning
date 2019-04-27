#!/bin/sh

NAME=exp_`date "+%m_%d_%H_%M_%S"`
ALGO=$1
EXP_FILE=$2
#NUM_WORKERS=$3
#
#if test -z "$NUM_WORKERS"
#then
#      WORKERS=''
#else
#      WORKERS=' --num_workers '"$NUM_WORKERS"
#fi
WORKERS=' --num_workers -1'
PORT=$3

tmux new -s $NAME -d
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m    # -m src.main master
tmux send-keys -t $NAME 'python -m cProfile -o output/profile_master.txt src/main.py master --master_socket_path /tmp/es_redis_master_'"$PORT"'.sock --algo '$ALGO' --exp_file '"$EXP_FILE"' 2>&1 | ts | tee ./output/master_p_outputfile.txt' C-m
tmux split-window -t $NAME
tmux send-keys -t $NAME '. src/scripts/local_env_setup.sh' C-m
tmux send-keys -t $NAME 'python -m cProfile -o output/profile_worker.txt src/main.py workers --master_port '"$PORT"' --master_host localhost --relay_socket_path /tmp/es_redis_relay_'"$PORT"'.sock --algo '"$ALGO"' '"$WORKERS"' 2>&1 | ts | tee ./output/worker_p_outputfile.txt' C-m
tmux a -t $NAME
