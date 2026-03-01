#!/bin/bash

if [ $# -ne 3 ]; then
    echo "Usage: $0 <p1_file_name> <p2_file_name> <num_games>"
    exit 1
fi

p1_file_name=$1
p2_file_name=$2
num_games=$3

output_sgf=$(echo $p1_file_name | sed 's/\/model\//-/g;s/.pt//g')_vs_$(echo $p2_file_name | sed 's/\/model\//-/g;s/.pt//g').sgf
if [ -f arena/$output_sgf ] && [ $(wc -l < arena/$output_sgf) -eq $num_games ]; then
    exit 1
else
    rm -f arena/$output_sgf
fi

# TODO: useD fight noD && fighting with different features
echo "P1: $p1_file_name, P2: $p2_file_name, Num Games: $num_games, Output SGF: arena/$output_sgf"
./build/phantomgo/minizero_phantomgo -conf_file phantomgo_evaluation.cfg \
    -conf_str iig_evaluation_player1_file_name=$p1_file_name:iig_evaluation_player2_file_name=$p2_file_name:iig_evaluation_discriminator1_file_name=$(echo $p1_file_name | sed 's/model/discriminator_model/g'):iig_evaluation_discriminator2_file_name=$(echo $p2_file_name | sed 's/model/discriminator_model/g') \
    -mode iig_arena > arena/$output_sgf &
pid=$!

while true; do
    p1_games=$(grep "P1\[$p1_file_name\]" arena/$output_sgf | wc -l)
    p2_games=$(grep "P1\[$p2_file_name\]" arena/$output_sgf | wc -l)
    echo "P1 games: $p1_games, P2 games: $p2_games"
    if [ $p1_games -ge $((num_games/2)) ] && [ $p2_games -ge $((num_games/2)) ]; then
        kill $pid
        tmp_file=$(mktemp)
        grep "P1\[$p1_file_name\]" arena/$output_sgf | head -n $((num_games/2)) > $tmp_file
        grep "P1\[$p2_file_name\]" arena/$output_sgf | head -n $((num_games/2)) >> $tmp_file
        mv $tmp_file arena/$output_sgf
        break
    fi
    sleep 5
done

