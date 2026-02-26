#!/bin/bash
set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("rabbit")
AblationCATEGORIES=(
    "albation-geometry-10000" "albation-geometry-10111" 
    "albation-geometry-11000" "albation-geometry-11100" 
    "albation-geometry-11110"
)


# Python 脚本路径
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/tests_ds/models/test_multiview_d_joint_sythetic.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="
    for ablation in "${AblationCATEGORIES[@]}"; do
        echo "  Processing ablation: $ablation"
        echo "============================="

        LOAD_PATH="$SynthoROOT/$category/$ablation/task.py"

        CUDA_VISIBLE_DEVICES=6 python "$SCRIPT_PATH" extract \
            --load "$LOAD_PATH"
    done
done

wait
