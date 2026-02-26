set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint"

# 定义类别
# SynthoCATEGORIES=("toy" "cat" "rabbit" "lego" "deer" "spidermanfight" "footballplayer")
SynthoCATEGORIES=("toy")

# Python 脚本路径
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/tests_ds/models/test_multiview_d_joint_sythetic.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"

    CUDA_VISIBLE_DEVICES=2 python "$SCRIPT_PATH" extract \
        --load "$LOAD_PATH"
done

# 等待所有后台任务完成
wait