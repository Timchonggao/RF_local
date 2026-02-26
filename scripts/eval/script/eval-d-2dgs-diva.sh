set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/diva_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("k1_double_punch" "penguin" "wolf")
# SynthoCATEGORIES=("dog")

MethodNmae="d2dgs"
# Python 脚本路径
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/gather_eval.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"
    BASELINE_VISUAL="$SynthoROOT/baselines/dynamic-2dgs/$category/test/ours_40000/renders"
    BASELINE_MESH="$SynthoROOT/baselines/dynamic-2dgs/$category/train/ours_40000"

    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_PATH" evaldiva \
        --load "$LOAD_PATH" \
        --baseline_visual "$BASELINE_VISUAL" \
        --baseline_mesh "$BASELINE_MESH" \
        --method_name "$MethodNmae"
done

# 等待所有后台任务完成
wait