import re

# 读取日志文件内容
with open("outputs/d_nvdiffrec_horse/test.1.2.1.1/log.txt", "r") as file:
    data = file.read()

# 匹配所有 Chamfer Distance 行和对应帧号
pattern = r"Frame (\d+): Chamfer Distance (GT next frame vs predicted|Pred next frame vs predicted|GT vs Pred): ([\deE\.\+-]+)"

# 临时存储结构：{frame_number: {"GT next": val, "Pred next": val, "GT pred": val}}
chamfer_data = {}

# 遍历匹配结果并填充数据结构
for match in re.finditer(pattern, data):
    frame = int(match.group(1))
    if 20 <= frame <= 219:
        distance_type = match.group(2)
        value = float(match.group(3))
        if frame not in chamfer_data:
            chamfer_data[frame] = {}
        if "GT next" in distance_type:
            chamfer_data[frame]["gt_next"] = value
        elif "Pred next" in distance_type:
            chamfer_data[frame]["pred_next"] = value
        elif "GT vs Pred" in distance_type:
            chamfer_data[frame]["gt_pred"] = value

# 提取有效帧（确保三类距离都有）
valid_frames = [v for v in chamfer_data.values() if len(v) == 3]

# 分别汇总三种距离
gt_next_vals = [v["gt_next"] for v in valid_frames]
pred_next_vals = [v["pred_next"] for v in valid_frames]
gt_pred_vals = [v["gt_pred"] for v in valid_frames]

# 打印平均值
print(f"Valid frames used: {len(valid_frames)}")
print(f"Average Chamfer Distance (GT next frame vs predicted): {sum(gt_next_vals)/len(gt_next_vals):.10f}")
print(f"Average Chamfer Distance (Pred next frame vs predicted): {sum(pred_next_vals)/len(pred_next_vals):.10f}")
print(f"Average Chamfer Distance (GT vs Pred): {sum(gt_pred_vals)/len(gt_pred_vals):.10f}")

