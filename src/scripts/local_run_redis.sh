#!/bin/sh

# https://stackoverflow.com/questions/6910378/how-can-i-stop-redis-server

PORT=$1

if test -z "$PORT"
then
      PORT=6379
else
      PORT="$PORT"
fi

tmux new -s redis -d
tmux send-keys -t redis 'redis-server redis_config/redis_master_'"$PORT"'.conf' C-m
tmux split-window -t redis
tmux send-keys -t redis 'redis-server redis_config/redis_local_mirror_'"$PORT"'.conf' C-m
tmux a -t redis
