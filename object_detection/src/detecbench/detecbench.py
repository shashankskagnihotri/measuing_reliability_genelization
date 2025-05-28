import os
from dataclasses import dataclass

import pandas as pd

# import mmdet
import tyro
from mmengine.config import Config
from mmengine.runner import Runner

from detecbench.attacks import BIM, FGSM, PGD, replace_val_loop
from detecbench.corruptions import AllCommonCorruptions, CommonCorruption, CommonCorruption3d


@dataclass
class IID:
    name = "iid"


def evaluate(
    task: PGD | FGSM | BIM | CommonCorruption | CommonCorruption3d | AllCommonCorruptions | IID,
    model_folder: str = "./models/RetinaNet_R-101-FPN",
    log_dir: str = "./logs",
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    retrieve_existing: bool = False,
    output_csv: str = "./results.csv",
) -> pd.DataFrame:
    """"""

    task_name = task.name

    if isinstance(task, AllCommonCorruptions):
        all_results = []

        for corruption_name in task.names:
            print(
                f"Running evaluation for corruption: {corruption_name}, severity: {task.severity}"
            )

            cc_task = CommonCorruption(
                name=corruption_name,  # pyright: ignore
                severity=task.severity,
                generate_missing=task.generate_missing,
            )
            result_df = evaluate(
                task=cc_task,
                model_folder=model_folder,
                log_dir=log_dir,
                wandb_project=wandb_project,
                wandb_entity=wandb_entity,
                retrieve_existing=retrieve_existing,
                output_csv=output_csv,
            )
            all_results.append(result_df)

        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            print(df)
            return df
        return pd.DataFrame()

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

    is_cc = isinstance(task, (CommonCorruption))
    is_3dcc = isinstance(task, CommonCorruption3d)
    if is_cc:
        assert not isinstance(task, (PGD, FGSM, BIM, IID))
        if cfg.dataset_type == "VOCDataset":
            cfg.val_dataloader.dataset.img_subdir = f"cc/{task.name}/severity_{task.severity}/"
        elif cfg.dataset_type == "CocoDataset":
            cfg.val_dataloader.dataset.data_prefix.img = f"cc/{task.name}/severity_{task.severity}/"
    elif is_3dcc:
        assert not isinstance(task, (PGD, FGSM, BIM, IID))
        if cfg.dataset_type == "VOCDataset":
            cfg.val_dataloader.dataset.img_subdir = f"{task.name}/{task.severity}/"
        elif cfg.dataset_type == "CocoDataset":
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
                        },  # task attributes
                    },
                    "group": model_name,
                },
            )
        )

    runner = Runner.from_cfg(cfg)

    if isinstance(task, (PGD, FGSM, BIM)):
        replace_val_loop(attack=task, use_wandb=use_wandb)

    # Check if the results file exists and if this task+model combo is already there
    if os.path.exists(output_csv) and retrieve_existing:
        existing_df = pd.read_csv(output_csv)

        mask = (existing_df["task"] == task_name) & (
            existing_df["model"] == os.path.basename(model_folder)
        )

        task_params = {k: v for k, v in task.__dict__.items() if not k.startswith("_")}
        for param_name, param_value in task_params.items():
            col_name = f"task_{param_name}"
            if col_name in existing_df.columns:
                mask = mask & (existing_df[col_name] == param_value)

        matching_results = existing_df[mask]
        assert isinstance(matching_results, pd.DataFrame)

        if not matching_results.empty:
            print(matching_results)
            return matching_results

    metrics = runner.val()

    # Save metrics
    if not metrics:
        df = pd.DataFrame()
    elif isinstance(metrics, dict) and all(isinstance(v, (int, float)) for v in metrics.values()):
        df = pd.DataFrame([metrics])  # Wrap in a list to create a single-row DataFrame
    else:
        df = pd.DataFrame(metrics)

    # Add model and task info if not empty
    if not df.empty:
        model_name = os.path.basename(model_folder)
        df["task"] = task_name
        df["model"] = model_name

    if isinstance(task, (PGD, FGSM, BIM)):
        df["iteration"] = df.index  # first row is unperturbed (0), then iterations
    # add task-specific parameters
    task_params = {k: v for k, v in task.__dict__.items() if not k.startswith("_")}
    for param_name, param_value in task_params.items():
        df[f"task_{param_name}"] = param_value

    print(df)
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        combined_df = pd.concat([existing_df, df], ignore_index=True)
        combined_df.to_csv(output_csv, index=False)
    else:
        df.to_csv(output_csv, index=False)

    return df


def main():
    tyro.cli(evaluate)


if __name__ == "__main__":
    main()
