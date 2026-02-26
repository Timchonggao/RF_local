set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("dog" "k1_double_punch" "penguin" "wolf")
# SynthoCATEGORIES=("dog")


SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/gather_eval.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"

    CUDA_VISIBLE_DEVICES=1 python "$SCRIPT_PATH" evaldiva \
        --load "$LOAD_PATH"
done

# 等待所有后台任务完成
wait