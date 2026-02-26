#!/bin/bash

set -e

# 所有目标目录（每个类别一个）
TARGETS=(
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/cat/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/deer/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/footballplayer/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/lego/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/rabbit/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/spidermanfight/test"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/blender_mv_d_joint/toy/test"
)

# 所有 baseline 路径
BASELINES=(
    # "/data3/gaochong/project/RadianceFieldStudio/outputs/baselines/atgs"
    # "/data3/gaochong/project/RadianceFieldStudio/outputs/baselines/deformable-3dgs"
    # "/data3/gaochong/project/RadianceFieldStudio/outputs/baselines/dg-mesh"
    "/data3/gaochong/project/RadianceFieldStudio/outputs/dynamic-2dgs"
    # "/data3/gaochong/project/RadianceFieldStudio/outputs/baselines/grid4d"
    # "/data3/gaochong/project/RadianceFieldStudio/outputs/baselines/sc-gs"
)

for TARGET in "${TARGETS[@]}"; do
    echo "处理目标目录: $TARGET"

    # 提取类别名（倒数第二层目录名）
    CATEGORY=$(basename "$(dirname "$TARGET")")
    echo "类别名: $CATEGORY"
    # 特殊处理类别名
    if [ "$CATEGORY" == "footballplayer" ]; then
        CATEGORY="football_player"
    fi
    if [ "$CATEGORY" == "spidermanfight" ]; then
        CATEGORY="spiderman_fight"
    fi
    echo "处理后的类别名: $CATEGORY"

    # 目标 baselines 路径
    TARGET_BASE="$TARGET/baselines"

    # 先删除再创建
    # rm -rf "$TARGET_BASE"
    # mkdir -p "$TARGET_BASE"

    # 遍历所有 baselines
    for BASE in "${BASELINES[@]}"; do
        echo "  处理 baseline: $BASE"

        SUBDIR_NAME=$(basename "$BASE")
        DEST="$TARGET_BASE/$SUBDIR_NAME"
        mkdir -p "$DEST"

        # 找到含有类别名的子目录
        CAT_DIR=$(find "$BASE" -maxdepth 1 -type d -name "*$CATEGORY*" | head -n 1)

        if [ -z "$CAT_DIR" ]; then
            echo "  未找到包含 $CATEGORY 的目录, 跳过"
            continue
        fi

        echo "  找到目录: $CAT_DIR"

        # 移动整个类别目录的内容到目标 baseline 子目录
        mv "$CAT_DIR"/* "$DEST/"
        echo "    已移动 $CAT_DIR 下的所有内容"

    done

    echo "✅ 完成类别 $CATEGORY 的处理"
done

echo "🎉 所有类别处理完成"
