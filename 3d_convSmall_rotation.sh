#!/bin/bash
python3 . --netname ../nets/convSmallRELU__DiffAI.pyt --geometric --geometric_config ./convSmall3d_rotation.txt --num_params 3 --dataset mnist --attack --skip_geom_ver --from_test 0 --num_tests 100 --geom_baseline 2 5 2 --geom_box 1 2 1 > 3d_rotation_convSmall_0_100.txt
