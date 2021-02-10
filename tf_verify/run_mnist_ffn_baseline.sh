#!/bin/bash
all=(6 7 15 24 26 31 33 36 40 41 45 46 52 53 62 66 73 77 78 92 96 98)
for i in "${all[@]}"
do
    echo $i
    python3 clever_wolf_main.py --netname ../nets/mnist_relu_9_200.tf --epsilon 0.045 --dataset mnist --max_cuts 0 --image_num $i --obox_init -100 --seed 42 --baseline 2>'err.txt' >"ffn_9_200_baseline_$i.txt"
done


