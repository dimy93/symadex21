#!/bin/bash
python3 . --netname ../nets/convSmallRELU_cifar__DiffAI.pyt --geometric --geometric_config  ./3d_rotation_cifar.txt --num_params 3 --dataset cifar10 --attack --skip_geom_ver --from_test 0 --num_tests 100  --geom_baseline 5 2 2 --geom_box 2 1 1 > 3d_rotation_cifar_0_100.txt
