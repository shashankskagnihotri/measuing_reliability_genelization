import argparse
import json

from pathlib import Path
from mmengine.config import DictAction
from typing import Dict, Optional


def get_args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='MMSeg test (and eval) a model')
    parser.add_argument('--config', help='Train config file path')
    parser.add_argument('--checkpoint', help='Checkpoint file path')
    parser.add_argument('--work-dir', help='Directory to save evaluation results as JSON')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help=(
            'Override config options. Format: key=val. For lists use key="[a,b]" or key=a,b. '
            'For nested structures: key="[(a,b),(c,d)]". No spaces allowed.'
        )
    )
    args, _ = parser.parse_known_args()
    return args


def get_config_path(model_name: str, backbone: str, dataset: str, crop_size: str = None) -> Path:
    if backbone and 'resnet' in backbone:
        backbone = backbone.replace('resnet', 'r')

    architecture_config_path = Path(f"mmsegmentation/configs/{model_name}")
    if not architecture_config_path.is_dir():
        raise FileNotFoundError(f"No config directory found for model '{model_name}'")
    
    configs_in_dir = [config_path for config_path in architecture_config_path.iterdir() if config_path.stem.startswith(model_name)]
        
    # filter by backbone, dataset and crop_size if given
    configs_in_dir = [config_path for config_path in configs_in_dir if backbone in config_path.stem]
    configs_in_dir = [config_path for config_path in configs_in_dir if dataset in config_path.stem]
    if crop_size:
        configs_in_dir = [config_path for config_path in configs_in_dir if crop_size in config_path.stem]
    
    if not configs_in_dir:
        raise FileNotFoundError(f"No config found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'.")
    if len(configs_in_dir) > 1:
        print(f"Warning: multiple configs found. Using: {configs_in_dir[0].name}")

    return configs_in_dir[0]


def get_checkpoint_path(model_name: str, backbone: str, dataset: str, crop_size: str = None) -> Path:
    if 'resnet' in backbone:
        backbone = backbone.replace('resnet', 'r')

    # search for checkpoint file
    checkpoint_files_path = Path(f"checkpoints/{model_name}")
    if not checkpoint_files_path.is_dir():
        raise FileNotFoundError(f"No checkpoint directory for model '{model_name}'")

    # get pth files that start with same filename as config filename
    checkpoints_in_dir = checkpoint_files_path.iterdir()
    
    # filter by backbone, dataset and crop_size if given
    checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if backbone in checkpoint_path.stem]
    checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if dataset in checkpoint_path.stem]
    if crop_size:
        checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if crop_size in checkpoint_path.stem]

    if len(checkpoints_in_dir) == 0:
        raise FileNotFoundError(f"No checkpoint file found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'")
    elif len(checkpoints_in_dir) > 1:
        raise ValueError(f"Multiple checkpoints found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'")

    return checkpoints_in_dir[0]


def get_results(work_dir: str) -> Dict[str, Optional[float]]:
    work_dir = Path(work_dir)
    result_dirs = [d for d in work_dir.iterdir() if d.is_dir()]
    if not result_dirs:
        raise FileNotFoundError(f"No result directories found in {work_dir}")
    
    # the result should be in the lastest timestamped directory
    latest_dir = max(result_dirs, key=lambda d: d.name)
    json_path = latest_dir / f"{latest_dir.name}.json"
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    metrics = ['aAcc', 'mIoU', 'mAcc']
    results = {metric: data.get(metric) for metric in metrics}
    return results