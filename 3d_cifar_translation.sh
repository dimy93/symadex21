#!/bin/bash
python3 . --netname ../nets/convSmallRELU_cifar__DiffAI.pyt --geometric --geometric_config 3d_translation_cifar.txt --num_params 3 --dataset cifar10 --attack --skip_geom_ver --from_test 0 --num_tests 100 --geom_baseline 4 3 3 --geom_box 2 1 1 > 3d_translation_cifar_0_100.txt
