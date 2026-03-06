#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dir1> <dir2> ..."
    exit 1
fi

for dir in "$@"; do
    echo "$dir:"
    for file in $(ls -rt $dir/*.sgf); do
        tac $file \
            | grep -oP 'RE\[[10\.-]+\].*?(?=;B\[(?:[0-9]|[1-9][0-9])\])' \
            | sed 's/1.000000/1/g;s/OBS\[\]//g;s/EV\[\]//g;s/ //g' | sed 's/\[/ /g;s/\]/ /g' | awk -v file=$(echo $file | awk -F "/" '{ print $NF; }') 'BEGIN {
                total = 0;
                first_name = second_name = "";
            }{
                p1 = p2 = p1_dnn = p2_dnn = p1_pimc_repeat = p2_pimc_repeat = re = "";
                for(i=1;i<=NF;i+=2) {
                    key = $i;
                    value = $(i+1);
                    if(key == "P1" || key == "P2" || key == "P1_DNN" || key == "P2_DNN") {
                        split(value, arr, "/");
                        split(arr[length(arr)], arr2, ".");
                        split(arr2[1], arr3, "_");
                        if(key == "P1") {
                            p1 = arr3[length(arr3)];
                        } else if(key == "P2") {
                            p2 = arr3[length(arr3)];
                        } else if(key == "P1_DNN") {
                            p1_dnn = arr3[length(arr3)];
                        } else if(key == "P2_DNN") {
                            p2_dnn = arr3[length(arr3)];
                        }
                    } else if(key == "P1_PIMC_REPEAT") {
                        p1_pimc_repeat = value;
                    } else if(key == "P2_PIMC_REPEAT") {
                        p2_pimc_repeat = value;
                    } else if(key == "RE") {
                        re = value;
                    } else if(key == "SZ") {
                        board_size = value;
                    }
                }
                if(p1_dnn == "") { p1_dnn = p1; }
                if(p2_dnn == "") { p2_dnn = p2; }
                if(p1_pimc_repeat == "") { p1_pimc_repeat = "5"; }
                if(p2_pimc_repeat == "") { p2_pimc_repeat = "5"; }

                p1_name = "P"p1"_D"p1_dnn"_R"p1_pimc_repeat;
                p2_name = "P"p2"_D"p2_dnn"_R"p2_pimc_repeat;
                if(p1 > p2) { first_name = p1_name; }
                else if(p1 < p2) { first_name = p2_name; }
                else {
                    if(p1_dnn > p2_dnn) { first_name = p1_name; }
                    else if(p1_dnn < p2_dnn) { first_name = p2_name; }
                    else {
                        if(p1_pimc_repeat > p2_pimc_repeat) { first_name = p1_name; }
                        else if(p1_pimc_repeat < p2_pimc_repeat) { first_name = p2_name; }
                        else { first_name = p1_name; }
                    }
                }
                second_name = (first_name == p1_name) ? p2_name : p1_name;
                if(re == "1") {
                    BW[p1_name]++;
                    WL[p2_name]++;
                } else if(re == "-1") {
                    BL[p1_name]++;
                    WW[p2_name]++;
                } else {
                    BD[p1_name]++;
                    WD[p2_name]++;
                }
            } END {
                total = BW[first_name] + WW[first_name] + BL[first_name] + WL[first_name] + BD[first_name] + WD[first_name];
                totalB = BW[first_name] + BL[first_name] + BD[first_name];
                totalW = WW[first_name] + WL[first_name] + WD[first_name];
                wr = (BW[first_name] + WW[first_name] + (BD[first_name] + WD[first_name]) / 2) * 100 / total;
                wrB = (BW[first_name] + (BD[first_name]) / 2) * 100 / totalB;
                wrW = (WW[first_name] + (WD[first_name]) / 2) * 100 / totalW;

                printf "%6.2f/%3d %6.2f/%3d %6.2f/%3d   %-20s %-20s (%dx%d)   %s\n",
                        wr, total,
                        wrB, totalB,
                        wrW, totalW,
                        first_name, second_name, board_size, board_size, file;
                # print wr"/"total, wrB"/"totalB, wrW"/"totalW, first_name, second_name;
            }'
    done
    echo ""
done
