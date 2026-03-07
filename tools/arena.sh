#!/usr/bin/env bash

set -euo pipefail

if [ $# -le 12 ]; then
    echo "Usage: $0 <p1_exec_file> <p1_cfg_file> <p1_nn_name> <p1_d_name> <p1_pimc_count> <p2_exec_file> <p2_cfg_file> <p2_nn_name> <p2_d_name> <p2_pimc_count> <num_games> <board_size> <p1_tag> <p2_tag>"
    exit 1
fi

p1_exec_file=$1; shift;
p1_cfg_file=$1; shift;
p1_file_name=$1; shift;
p1_d_file_name=$1; shift;
p1_pimc_repeat=$1; shift;
p2_exec_file=$1; shift;
p2_cfg_file=$1; shift;
p2_file_name=$1; shift;
p2_d_file_name=$1; shift;
p2_pimc_repeat=$1; shift;
num_games=$1; shift;
board_size=$1; shift;
p1_tag=$1; shift;
p2_tag=$1; shift;

LOG_FILE="./arena/"
LOG_FILE+="$(echo $p1_file_name | cut -d "/" -f 1)_"
LOG_FILE+="$(echo $p1_file_name | awk -F "_" '{ print $NF; }' | cut -d "." -f 1)_"
LOG_FILE+="$(echo $p1_d_file_name | awk -F "_" '{ print $NF; }' | cut -d "." -f 1)_"
LOG_FILE+="$p1_pimc_repeat""_"
LOG_FILE+="$p1_tag"
LOG_FILE+="_vs_"
LOG_FILE+="$(echo $p2_file_name | cut -d "/" -f 1)_"
LOG_FILE+="$(echo $p2_file_name | awk -F "_" '{ print $NF; }' | cut -d "." -f 1)_"
LOG_FILE+="$(echo $p2_d_file_name | awk -F "_" '{ print $NF; }' | cut -d "." -f 1)_"
LOG_FILE+="$p2_pimc_repeat""_"
LOG_FILE+="$p2_tag"
LOG_FILE+=".sgf"

echo "P1"
echo -e " - exec: $p1_exec_file"
echo -e " - cfg: $p1_cfg_file"
echo -e " - nn: $p1_file_name"
echo -e " - dnn: $p1_d_file_name"
echo -e " - pimc: $p1_pimc_repeat"
echo -e " - tag: $p1_tag"

echo "P2"
echo -e " - exec: $p2_exec_file"
echo -e " - cfg: $p2_cfg_file"
echo -e " - nn: $p2_file_name"
echo -e " - dnn: $p2_d_file_name"
echo -e " - pimc: $p2_pimc_repeat"
echo -e " - tag: $p2_tag"

echo ""
echo "Num Games: $num_games"
echo "Output file: $LOG_FILE"

if [ -f $LOG_FILE ] && [ $(wc -l < $LOG_FILE) -ge $num_games ]; then
  exit 1
else
  rm -f $LOG_FILE
  touch "$LOG_FILE"
fi

A_CMD=(
  stdbuf -o0 -e0
  $p1_exec_file
  -conf_file $p1_cfg_file
  -conf_str "iig_player_file_name=$p1_file_name:iig_discriminator_file_name=$p1_d_file_name:actor_pimc_repeat=$p1_pimc_repeat:iig_arena_tag=$p1_tag:env_board_size=$board_size"
  -mode iig_arena
)

B_CMD=(
  stdbuf -o0 -e0
  $p2_exec_file
  -conf_file $p2_cfg_file
  -conf_str "iig_player_file_name=$p2_file_name:iig_discriminator_file_name=$p2_d_file_name:actor_pimc_repeat=$p2_pimc_repeat:iig_arena_tag=$p2_tag:env_board_size=$board_size"
  -mode iig_arena
)

zero_num_parallel_games=$(grep "zero_num_parallel_games" $p1_cfg_file | cut -d ' ' -f 1 | cut -d '=' -f 2)

CMDS="set_genmove"
for ((i=0; i<zero_num_parallel_games; i++)); do
  CMDS+=" $i"
done
INIT_CMDS=("$CMDS" "genmove")

TMP_DIR="$(mktemp -d)"
A_IN_FIFO="$TMP_DIR/a_in.fifo"
B_IN_FIFO="$TMP_DIR/b_in.fifo"
B_OUT_FIFO="$TMP_DIR/b_out.fifo"

mkdir -p "$TMP_DIR"
mkfifo "$A_IN_FIFO" "$B_IN_FIFO" "$B_OUT_FIFO"

filter_stderr() {
  local who="$1"
  while IFS= read -r line; do
    if [[ "$line" == \[GAME_LOG\]* ]]; then
      printf '%s\n' "$line" >> "$LOG_FILE"
    fi
    printf '[%s] %s\n' "$who" "$line" >&2
  done
}

# Keep FIFOs open from shell side to avoid open/close races and accidental EOF
exec 3<>"$A_IN_FIFO"
exec 4<>"$B_IN_FIFO"
exec 5<>"$B_OUT_FIFO"

# Merge: initial commands first, then B stdout -> A stdin
{
  for cmd in "${INIT_CMDS[@]}"; do
    printf '%s\n' "$cmd"
  done
  cat < "$B_OUT_FIFO"
} > "$A_IN_FIFO" &
MERGE_PID=$!

# Start A: stdin <- A_IN_FIFO, stdout -> B_IN_FIFO
"${A_CMD[@]}" \
  < "$A_IN_FIFO" \
  > "$B_IN_FIFO" \
  2> >(
    filter_stderr "A"
  ) &
A_PID=$!

# Start B: stdin <- B_IN_FIFO, stdout -> B_OUT_FIFO
"${B_CMD[@]}" \
  < "$B_IN_FIFO" \
  > "$B_OUT_FIFO" \
  2> >(
    filter_stderr "B"
  ) &
B_PID=$!

cleanup() {
  trap - INT TERM EXIT

  echo "Cleaning up..." >&2

  [[ -n "${MERGE_PID:-}" ]] && kill "$MERGE_PID" 2>/dev/null || true
  [[ -n "${A_PID:-}"     ]] && kill "$A_PID"     2>/dev/null || true
  [[ -n "${B_PID:-}"     ]] && kill "$B_PID"     2>/dev/null || true

  pkill -TERM -P $$ 2>/dev/null || true
  sleep 0.2
  pkill -KILL -P $$ 2>/dev/null || true

  exec 3>&- 4>&- 5>&- 2>/dev/null || true

  rm -rf "$TMP_DIR"
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM
trap cleanup EXIT

while true; do
    total_games=$(wc -l < "$LOG_FILE" 2>/dev/null || echo 0)
    p1_b_games=$(grep "B_NN\[$p1_file_name\]" $LOG_FILE | grep "B_DNN\[$p1_d_file_name\]" | grep "B_PIMC_REPEAT\[$p1_pimc_repeat\]" | grep "B_TAG\[$p1_tag\]" | wc -l || true)
    p1_w_games=$(grep "W_NN\[$p1_file_name\]" $LOG_FILE | grep "W_DNN\[$p1_d_file_name\]" | grep "W_PIMC_REPEAT\[$p1_pimc_repeat\]" | grep "W_TAG\[$p1_tag\]" | wc -l || true)
    if [ $p1_b_games -ge $(($num_games/2)) ] && [ $p1_w_games -ge $(($num_games/2)) ]; then
        kill "$A_PID" "$B_PID" 2>/dev/null || true
        wait "$A_PID" "$B_PID" 2>/dev/null || true
        sleep 1

        tmp_file=$(mktemp)
        echo "tmp_file: $tmp_file"
        grep "B_NN\[$p1_file_name\]" $LOG_FILE | grep "B_DNN\[$p1_d_file_name\]" | grep "B_PIMC_REPEAT\[$p1_pimc_repeat\]" | grep "B_TAG\[$p1_tag\]" | head -n $(($num_games/2)) > $tmp_file || true
        grep "W_NN\[$p1_file_name\]" $LOG_FILE | grep "W_DNN\[$p1_d_file_name\]" | grep "W_PIMC_REPEAT\[$p1_pimc_repeat\]" | grep "W_TAG\[$p1_tag\]" | head -n $(($num_games/2)) >> $tmp_file || true
        echo "mv $tmp_file $LOG_FILE"
        mv $tmp_file $LOG_FILE
        echo "after mv"
        break
    fi
    sleep 5
done