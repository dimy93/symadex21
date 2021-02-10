#!/bin/bash
python3 . --netname ../nets/convBigRELU__DiffAI.pyt --geometric --geometric_config ./4d_rotation.txt --num_params 4 --dataset mnist --attack --skip_geom_ver --from_test 0 --num_tests 100 --geom_box 2 1 1 1 --geom_baseline 3 2 2 2 > 4d_rotation_convBig_0_100.txt
