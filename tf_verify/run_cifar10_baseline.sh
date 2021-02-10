#!/bin/bash
attackable=(3 6 7 8 9 18 21 26 30 32 41 55 56 63 65 66 70 74 84 86 96 98 99)
for i in ${attackable[@]}
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/convSmallRELU__cifar10_Point.pyt --max_cuts 0 --epsilon 0.006 --dataset cifar10 --image_num $i --seed 42 --obox_init -10000 --baseline 2>'err.txt' >"convSmall_cifar_baseline_$i.txt"
done
