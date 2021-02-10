#!/bin/bash
python3 . --netname ../nets/convBigRELU__DiffAI.pyt --geometric --geometric_config 3d_translation.txt --num_params 3 --dataset mnist --attack --skip_geom_ver --from_test 0 --num_tests 100 --geom_box 2 1 1 --geom_baseline 4 2 2 --c 0.75 > 3d_translation_convBig_0_100_075.txt
