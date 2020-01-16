# 
# Created by ZhangYuyang on 2020/1/15
#
# 专用于预处理MegaDepth数据集
import argparse

from data_utils.megadepth_dataset import MegaDetphDatasetCreator

args = argparse.ArgumentParser(description="MegaDepthDataset Preprocess.")
args.add_argument("--scenes_ratio", type=float, default=1.0)
args.add_argument("--pairs_per_scene", type=float, default=0.5)
args.add_argument("--base_path", type=str, default="/data/MegaDepthOrder")
args.add_argument("--scene_info_path", type=str, default="/data/MegaDepthOrder/scene_info")
args.add_argument("--dataset_type", type=str, default="train")
params = args.parse_args()

if params.dataset_type == "train":
    creator = MegaDetphDatasetCreator(
        base_path=params.base_path,
        scene_info_path=params.scene_info_path,
        scenes_ratio=params.scenes_ratio,
        pairs_per_scene=params.pairs_per_scene,
        train=True,
    )
    creator.build_dataset()
elif params.dataset_type == "val":
    creator = MegaDetphDatasetCreator(
        base_path=params.base_path,
        scene_info_path=params.scene_info_path,
        scenes_ratio=params.scenes_ratio,
        pairs_per_scene=params.pairs_per_scene,
        train=False,
    )
    creator.build_dataset()
else:
    print("Unrecognized dataset_type:%s" % params.dataset_type)
    assert False



