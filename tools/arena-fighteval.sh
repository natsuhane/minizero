#!/bin/bash

if [ $# -ne 6 ]; then
    echo "Usage: $0 <dir> <start_iter> <end_iter> <interval> <result_dir> <board_size>"
    exit 1
fi

dir=$1
start_iter=$2
end_iter=$3
interval=$4
result_dir=$5
board_size=$6

trap "echo 'Stopping fighteval...'; kill 0" SIGINT SIGTERM

mkdir -p $result_dir
for ((iter=start_iter; iter<end_iter; iter+=interval)); do
    modelA=$dir/model/weight_iter_$iter.pt
    modelB=$dir/model/weight_iter_$((iter+interval)).pt
    if [ ! -f $modelA ] || [ ! -f $modelB ]; then
        continue
    fi

    ./tools/arena.sh $modelA $modelB 500 $result_dir $board_size
done
