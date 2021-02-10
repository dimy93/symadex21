#!/bin/bash
python3 . --netname ../nets/convSmallRELU__DiffAI.pyt --geometric --geometric_config ./3d_convSmall_translation.txt --num_params 3 --dataset mnist --attack --skip_geom_ver --from_test 0 --num_tests 100 --geom_box 2 1 1 --geom_baseline 4 3 3 > 3d_translation_convSmall_0_100.txt
