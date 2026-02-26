from rfstudio_ds.engine.experiment import DS_Experiment
from pathlib import Path
import torch


def test_experiment():
    # 测试初始化
    test_name = "test_experiment"
    test_output_dir = Path('outputs/test_ds_outputs/engine/experiment')
    experiment = DS_Experiment(name=test_name, output_dir=test_output_dir)
    experiment.parse_log_auto(log_file="/data3/gaochong/project/RadianceFieldStudio/outputs/d_nvdiffrec_girlwalk/debug_parse_log/log.txt")
    # print(f"Timestamp: {experiment.timestamp}")
    # print(f"base_path: {experiment.base_path}")
    # print(f"log_path: {experiment.log_path}")
    # print(f"dump_path: {experiment.dump_path}")
    # print(f"output_dir: {experiment.output_dir}")
    
    # # 测试 logging
    # experiment.log("Test log entry.")

    # # 测试 image_dumping
    # image = torch.rand(100, 100, 3)  # 创建一个随机的形状为 (H, W, 3) 的图像张量
    # experiment.dump_image(subfolder="images", index=0, image=image)

    # # 测试 video_dumping
    # images = [torch.rand(100, 100, 3) for _ in range(100)]  # 创建 10 个随机的形状为 (H, W, 3) 的图像张量
    # experiment.dump_images2video(subfolder="videos", images=images, target_mb=0.1, index=0)


if __name__ == "__main__":
    test_experiment()
