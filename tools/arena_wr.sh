#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <file1> <file2> ..."
    exit 1
fi

grep_str=""
for str in "$@"; do
    grep_str+="$str\|"
done
grep_str=${grep_str%\\|}

for file in $(ls -rt arena/* | grep "$grep_str"); do
    p1_folder=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $1; }' | awk -F "_" '{ for(i=1;i<=NF-5;++i) { printf $i"_"; } print $(NF-4); }')
    p2_folder=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $2; }' | awk -F "_" '{ for(i=1;i<=NF-5;++i) { printf $i"_"; } print $(NF-4); }')
    p1_nn=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $1; }' | awk -F "_" -v folder=$p1_folder '{ print folder"/model/weight_iter_"$(NF-3)".pt"; }')
    p2_nn=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $2; }' | awk -F "_" -v folder=$p2_folder '{ print folder"/model/weight_iter_"$(NF-3)".pt"; }')
    p1_dnn=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $1; }' | awk -F "_" -v folder=$p1_folder '{ if($(NF-2)!="") {print folder"/discriminator_model/weight_iter_"$(NF-2)".pt";} }')
    p2_dnn=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $2; }' | awk -F "_" -v folder=$p2_folder '{ if($(NF-2)!="") {print folder"/discriminator_model/weight_iter_"$(NF-2)".pt";} }')
    p1_pimc=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $1; }' | awk -F "_" '{ print $(NF-1); }')
    p2_pimc=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $2; }' | awk -F "_" '{ print $(NF-1); }')
    p1_tag=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $1; }' | awk -F "_" '{ print $NF; }')
    p2_tag=$(echo $file | cut -d '/' -f2 | cut -d '.' -f1 | awk -F "_vs_" '{ print $2; }' | awk -F "_" '{ print $NF; }')
    file_with_color=$(echo $file | awk -F "/" '{ print $NF; }' | sed -E 's/(phantomgo_)([^_]+)(_gaz)/\1\x1b[1;31m\2\x1b[0m\3/;s/_([0-9]+_[0-9]+_[0-9]+)_vs_/_\x1b[1;31m\1\x1b[0m_vs_/;s/_([0-9]+_[0-9]+_[0-9]+)\.sgf/_\x1b[1;31m\1\x1b[0m.sgf/')
    tac $file \
        | grep -oP 'RE\[[10\.-]+\].*?(?=;B\[(?:[0-9]|[1-9][0-9])\])' \
        | sed 's/1.000000/1/g;s/OBS\[\]//g;s/EV\[\]//g;s/B_TAG\[\]//g;s/W_TAG\[\]//g;s/B_DNN\[\]//g;s/W_DNN\[\]//g;;s/ //g' | sed 's/\[/ /g;s/\]/ /g' | awk \
            -v file=$file_with_color \
            -v first_name=$p1_nn$p1_dnn$p1_pimc$p1_tag \
            -v second_name=$p2_nn$p2_dnn$p2_pimc$p2_tag 'BEGIN {
            total = 0;
        }{
            p1 = p2 = p1_dnn = p2_dnn = p1_pimc_repeat = p2_pimc_repeat = p1_tag = p2_tag = re = "";
            for(i=1;i<=NF;i+=2) {
                key = $i;
                value = $(i+1);
                if(key == "B_NN") {
                    p1 = value;
                } else if(key == "W_NN") {
                    p2 = value;
                } else if(key == "B_DNN") {
                    p1_dnn = value;
                } else if(key == "W_DNN") {
                    p2_dnn = value;
                } else if(key == "B_PIMC_REPEAT") {
                    p1_pimc_repeat = value;
                } else if(key == "W_PIMC_REPEAT") {
                    p2_pimc_repeat = value;
                } else if(key == "B_TAG") {
                    p1_tag = value;
                }else if(key == "W_TAG") {
                    p2_tag = value;
                } else if(key == "RE") {
                    re = value;
                } else if(key == "SZ") {
                    board_size = value;
                }
            }
            if(p1_dnn == "") { p1_dnn = ""; }
            if(p2_dnn == "") { p2_dnn = ""; }
            if(p1_pimc_repeat == "") { p1_pimc_repeat = "5"; }
            if(p2_pimc_repeat == "") { p2_pimc_repeat = "5"; }
            if(p1_tag == "") { p1_tag = ""; }
            if(p2_tag == "") { p2_tag = ""; }

            p1_name = p1""p1_dnn""p1_pimc_repeat""p1_tag;
            p2_name = p2""p2_dnn""p2_pimc_repeat""p2_tag;
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

            printf "%6.2f/%3d %6.2f/%3d %6.2f/%3d  %s\n",
                    wr, total,
                    wrB, totalB,
                    wrW, totalW, file;
            # print wr"/"total, wrB"/"totalB, wrW"/"totalW, first_name, second_name;
        }'
done
