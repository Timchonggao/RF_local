set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("toy" "cat" "rabbit" "lego" "deer" "spidermanfight" "footballplayer")
# SynthoCATEGORIES=("toy")

# Python 脚本路径
# SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/eval-scgs.py"
MethodNmae="sc-gs"
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/gather_eval.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"
    BASELINE_VISUAL="$SynthoROOT/$category/baselines/sc-gs/test/ours_80000/renders"
    BASELINE_MESH="$SynthoROOT/baselines_mesh/sc-gs/$category/train/ours_80000/mesh"

    CUDA_VISIBLE_DEVICES=0 python "$SCRIPT_PATH" evalsyn \
        --load "$LOAD_PATH" \
        --baseline_visual "$BASELINE_VISUAL" \
        --baseline_mesh "$BASELINE_MESH" \
        --method_name "$MethodNmae"
done

# 等待所有后台任务完成
wait