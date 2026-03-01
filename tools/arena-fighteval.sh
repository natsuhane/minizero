#!/bin/bash

if [ $# -ne 4 ]; then
    echo "Usage: $0 <dir> <start_iter> <end_iter> <interval>"
    exit 1
fi

dir=$1
start_iter=$2
end_iter=$3
interval=$4

for ((iter=start_iter; iter<end_iter; iter+=interval)); do
    modelA=$dir/model/weight_iter_$iter.pt
    modelB=$dir/model/weight_iter_$((iter+interval)).pt
    if [ ! -f $modelA ] || [ ! -f $modelB ]; then
        continue
    fi

    ./tools/arena.sh $modelA $modelB 500
done
