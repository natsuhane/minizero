#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <p1_file_name> <p2_file_name> <num_games> <result_dir> <board_size>"
    exit 1
fi

p1_file_name=$1
p2_file_name=$2
num_games=$3
result_dir=$4
board_size=$5

output_sgf=$(echo $p1_file_name | sed 's/\/model\//-/g;s/.pt//g')_vs_$(echo $p2_file_name | sed 's/\/model\//-/g;s/.pt//g').sgf
if [ -f $result_dir/$output_sgf ] && [ $(wc -l < $result_dir/$output_sgf) -eq $num_games ]; then
    exit 1
else
    rm -f $result_dir/$output_sgf
fi

# TODO: useD fight noD && fighting with different features
echo "P1: $p1_file_name, P2: $p2_file_name, Num Games: $num_games, Output SGF: $result_dir/$output_sgf"
./build/phantomgo/minizero_phantomgo -conf_file phantomgo_evaluation.cfg \
    -conf_str iig_evaluation_player1_file_name=$p1_file_name:iig_evaluation_player2_file_name=$p2_file_name:iig_evaluation_discriminator1_file_name=$(echo $p1_file_name | sed 's/model/discriminator_model/g'):iig_evaluation_discriminator2_file_name=$(echo $p2_file_name | sed 's/model/discriminator_model/g'):env_board_size=$board_size \
    -mode iig_arena > $result_dir/$output_sgf &
pid=$!

cleanup() {
    echo ""
    echo "Caught Ctrl+C, killing process $pid ..."
    
    # Kill the background process if still running
    if ps -p $pid > /dev/null 2>&1; then
        kill $pid
        wait $pid 2>/dev/null
    fi
    
    echo "Cleanup done."
    exit 1
}

trap cleanup SIGINT

while true; do
    p1_games=$(grep "P1\[$p1_file_name\]" $result_dir/$output_sgf | wc -l)
    p2_games=$(grep "P1\[$p2_file_name\]" $result_dir/$output_sgf | wc -l)
    echo "P1 games: $p1_games, P2 games: $p2_games"
    if [ $p1_games -ge $((num_games/2)) ] && [ $p2_games -ge $((num_games/2)) ]; then
        kill $pid
        tmp_file=$(mktemp)
        grep "P1\[$p1_file_name\]" $result_dir/$output_sgf | head -n $((num_games/2)) > $tmp_file
        grep "P1\[$p2_file_name\]" $result_dir/$output_sgf | head -n $((num_games/2)) >> $tmp_file
        mv $tmp_file $result_dir/$output_sgf
        break
    fi
    sleep 5
done

