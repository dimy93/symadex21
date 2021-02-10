#!/bin/bash
python3 . --netname ../nets/convSmallRELU_cifar__DiffAI.pyt --geometric --geometric_config ./4d_rotation_cifar_config.txt --num_params 4 --dataset cifar10 --attack --skip_geom_ver --from_test 50 --num_tests 50 --geom_baseline 5 2 2 2 --geom_box 2 1 1 1 > 4d_rotation_cifar_0_100.txt
