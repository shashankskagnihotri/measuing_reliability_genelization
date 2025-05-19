import os
from dataclasses import dataclass

import mmdet
import tyro
from mmengine.config import Config
from mmengine.runner import Runner

from detecbench.attacks import BIM, FGSM, PGD, replace_val_loop
from detecbench.corruptions import CommonCorruption, CommonCorruption3d


@dataclass
class IID:
    """Independent and Identically Distributed (IID) evaluation."""

    pass


def evaluate(
    task: PGD | FGSM | BIM | CommonCorruption | CommonCorruption3d | IID,
    model_folder: str = "./models/RetinaNet_R-101-FPN",
    log_dir: str = "./logs",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
):
    """"""

    # Parse task type
    is_attack = False
    is_cc = False
    is_3dcc = False
    if isinstance(task, PGD):
        is_attack = True
        task_name = "PGD"
    elif isinstance(task, FGSM):
        is_attack = True
        task_name = "FGSM"
    elif isinstance(task, BIM):
        is_attack = True
        task_name = "BIM"
    elif isinstance(task, CommonCorruption):
        is_cc = True
        task_name = "CC"
    elif isinstance(task, CommonCorruption3d):
        is_3dcc = True
        task_name = "3DCC"
    elif isinstance(task, IID):
        task_name = "iid"
    else:
        raise NotImplementedError

    # Setup the configuration
    # Look for configuration (.py) and checkpoint (.pth) files in the folder
    config_files = [f for f in os.listdir(model_folder) if f.endswith(".py")]
    checkpoint_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]

    if not config_files:
        raise ValueError(f"No configuration (.py) file found in the folder: {model_folder}")
    if not checkpoint_files:
        raise ValueError(f"No checkpoint (.pth) file found in the folder: {model_folder}")

    # Use the first found config and checkpoint file
    config_path = os.path.join(model_folder, config_files[0])
    checkpoint_path = os.path.join(model_folder, checkpoint_files[0])

    print(f"Using configuration: {config_path}")
    print(f"Using checkpoint: {checkpoint_path}")

    cfg = Config.fromfile(config_path)
    cfg.work_dir = log_dir
    cfg.load_from = checkpoint_path

    if is_cc:
        assert not isinstance(task, (PGD, FGSM, BIM, IID))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/severity_{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/severity_{task.severity}/"
    elif is_3dcc:
        assert not isinstance(task, (PGD, FGSM, BIM, IID))
        if task.dataset == "Pascal":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/{task.severity}/"
        elif task.dataset == "Coco":
            cfg.val_dataloader.dataset.data_prefix.img = f"{task.name}/{task.severity}/"

    # Setup optional wandb logging
    use_wandb = wandb_project is not None and wandb_entity is not None
    if use_wandb:
        cfg.default_hooks.visualization.draw = True
        cfg.default_hooks.visualization.interval = 1000
        model_name = model_folder.split("/")[-1].split(".")[0]
        cfg.visualizer.vis_backends = dict(
            dict(
                type="WandbVisBackend",
                init_kwargs={
                    "project": wandb_project,
                    "entity": wandb_entity,
                    "config": {
                        "task_name": task_name,
                        "model": "model",
                        "checkpoint": checkpoint_path,
                        **{
                            k: v for k, v in task.__dict__.items() if not k.startswith("_")
                        },  # attack attributes
                    },
                    "group": model_name,
                },
            )
        )

    runner = Runner.from_cfg(cfg)

    if is_attack and not isinstance(task, (CommonCorruption, CommonCorruption3d, IID)):
        replace_val_loop(attack=task, use_wandb=use_wandb)

    runner.val()


def main():
    tyro.cli(evaluate)


if __name__ == "__main__":
    main()
