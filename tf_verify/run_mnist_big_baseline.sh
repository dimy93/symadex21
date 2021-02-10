#!/bin/bash
all=(7 12 15 18 20 25 29 31 40 41 45 58 59 62 65 73 78 96)
for i in "${all[@]}"
do
echo $i
    taskset -c 0-16 python3 clever_wolf_main.py --netname ../nets/ConvBig__Point_mnist.pyt --epsilon 0.05 --max_cuts 0 --dataset mnist --image_num $i --obox_init -10 --seed 42 --nowolf --baseline 2>'err.txt' >"convBig_baseline_$i.txt"
    #t="ConvBig__Point_mnist_"$i"_class_*" 
    #for value in $t; do python3 viz.py $value; done
done

