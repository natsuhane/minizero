#!/bin/bash

if [ $# -lt 1 ]; then
    echo "Usage: $0 <dir1> <dir2> ..."
    exit 1
fi

for dir in "$@"; do
    echo "$dir:"
    cat $dir/*.sgf \
        | grep -o "RE\[[0-9.-]*\]OBS\[\]SZ\[[0-9]\]KM\[1.000000\]EV\[\]T\[[0-9/ :.]*\]P2\[[a-zA-Z/._0-9-]*\]P1\[[a-zA-Z/._0-9-]*\]" \
        | sed 's/1.000000/1/g;s/OBS\[\]SZ\[[0-9]\]KM\[1\]EV\[\]//g;s/T\[[0-9/ :.]*\]//g' | sort | uniq -c | sed 's/P2\[/ /g;s/\]P1\[/ /g;s/\]//g;s/RE\[//g' \
        | awk '{ split($3,a,"/"); split($4,b,"/"); print $1,$2,a[3],b[3]; }' \
        | awk '{ if($2==1) { print $1,$4,$3; } else if($2==-1) { print $1,$3,$4; } }' \
        | sort -k2 -V \
        | awk '{ s+=$1; if(c%2==1) { print s,$2,$3; s = 0; } c++; }' \
        | awk '{ if(c%2==1) { print $1*100/(s+$1)"/"(s+$1),$2,$3; s = 0; } s=$1; c++; }'
    echo ""
done
