set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/cmupanonic_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("cello1" "hanggling_b2" "ian3" "pizza1")
# SynthoCATEGORIES=("band1")

# Python 脚本路径
# SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/eval-deformable-3dgs.py"
MethodNmae="deformable-3dgs"
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/scripts/eval/gather_eval.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    LOAD_PATH="$SynthoROOT/$category/test/task.py"
    BASELINE_VISUAL="$SynthoROOT/0_baselines/deformable-3dgs/$category/train/ours_40000/renders"
    BASELINE_MESH="$SynthoROOT/0_baselines/deformable-3dgs/$category/train/ours_40000/mesh"

    CUDA_VISIBLE_DEVICES=6 python "$SCRIPT_PATH" evalcmu \
        --load "$LOAD_PATH" \
        --baseline_visual "$BASELINE_VISUAL" \
        --baseline_mesh "$BASELINE_MESH" \
        --method_name "$MethodNmae"
done

# 等待所有后台任务完成
wait