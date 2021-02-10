#!/bin/bash
attackable=(6 8 9 11 15 18 20 24 33 38 45 48 61 62 65 66 73 78 92 95 96)
for i in ${attackable[@]}
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/convSmallRELU__Point.pyt --epsilon 0.12 --dataset mnist --image_num $i --seed 42 --max_cuts 0 --obox_init -10000 2>'err.txt' >"convSmall_$i.txt"
    t="convSmallRELU__Point_"$i"_class_*"
    for value in $t; do python3 viz.py $value; done
done
