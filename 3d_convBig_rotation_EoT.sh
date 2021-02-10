#!/bin/bash
python3 . --netname ../nets/convBigRELU__DiffAI.pyt --geometric --geometric_config ./3d_rotation.txt --num_params 3 --dataset mnist --attack --skip_geom_ver --from_test 0 --num_tests 100 --eot > 3d_rotation_convBig_0_100_eot.txt
