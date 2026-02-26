set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/realobject_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("dog" "k1_double_punch" "penguin" "wolf")
# SynthoCATEGORIES=()

# Python 脚本路径
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/tests_ds/models/test_multiview_d_joint_diva.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"

    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_PATH" eval \
        --load "$LOAD_PATH"
done

# 等待所有后台任务完成
wait