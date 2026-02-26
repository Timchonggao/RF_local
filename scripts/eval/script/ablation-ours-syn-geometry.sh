set -e

# 实验根目录
SynthoROOT="/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint"

# 定义类别
SynthoCATEGORIES=("toy" "cat" "rabbit" "lego" "deer" "spidermanfight" "footballplayer")
# SynthoCATEGORIES=("toy")

# Python 脚本路径
SCRIPT_PATH="/data3/gaochong/project/RadianceFieldStudio/tests_ds/models/test_multiview_d_joint_sythetic.py"

for category in "${SynthoCATEGORIES[@]}"; do
    echo "============================="
    echo "  Processing category: $category"
    echo "============================="

    CUDA_VISIBLE_DEVICES=3 python "$SCRIPT_PATH" $category --experiment.timestamp albation-nopbr \
        --model.shader_type direct --trainer.num_steps_per_val_pbr_attr 10000
done

# 等待所有后台任务完成
wait