#!/bin/bash
while ((0<1))
do
    g0=$(nvidia-smi  -i 0 -q --display=MEMORY | grep Free | head -1 | awk '{print $(NF-1)}')

    mem=10000
    if [ $g0 -gt $mem ]
    then
        echo 'using gpu 0'
        python tools/train.py
        break
    fi

    sleep 2s
    echo $g0
done