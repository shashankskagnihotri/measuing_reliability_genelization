import pandas as pd
import os
import yaml

from pathlib import Path
from mmseg.apis import MMSegInferencer

from .semseg_wrapper import SemSegWrapper


common_corruptions = [
    'gaussian_noise',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'jpeg_compression',
] 
    
attacks = [
    'pgd',
    'segpgd',
    'cospgd',
]


def load_model(model_name: str, backbone: str = None, dataset: str = None, crop_size: str = None,
               load_without_weights = False):
    """
    Load and return model specified by parameters
    - model_name (str): either just architecture name like 'segformer' or whole config filename like 'segformer_mit-b0_8xb2-160k_ade20k-512x512'
    - (optional) backbone (str): for example 'mit-b0'
    - (optional) dataset (str): for example 'ade20k'
    - (optional) crop_size (str): for example '512x512'

    - load_without_weights (bool) default False: if True, no checkoint file will be searched and model will be loaded withou weights
    """
    model_name = model_name.lower()
    backbone = backbone.lower()
    if 'resnet' in backbone:
        backbone = backbone.replace('resnet', 'r')
    dataset = dataset.lower()

    model_architecture = model_name.split("_")[0]
    architecture_config_path = Path(f"mmsegmentation/configs/{model_architecture}")
    if not architecture_config_path.is_dir():
        raise FileNotFoundError(f"No config for model with architecture name '{model_architecture}' existent!")
    
    # if model_name contains underscore --> expect whole config filename
    if len(model_name.split("_")) > 1:
        config_path = architecture_config_path / Path(model_name+".py")
        if not config_path.is_file():
            raise FileNotFoundError(f"No config found for model {model_name}")
    else:
        configs_in_dir = [config_path for config_path in architecture_config_path.iterdir() if config_path.stem.startswith(model_architecture)]
        
        # filter by backbone, dataset and crop_size if given
        if backbone:
            configs_in_dir = [config_path for config_path in configs_in_dir if backbone in config_path.stem]
        if dataset:
            configs_in_dir = [config_path for config_path in configs_in_dir if dataset in config_path.stem]
        if crop_size:
            configs_in_dir = [config_path for config_path in configs_in_dir if crop_size in config_path.stem]
        """
        if len(configs_in_dir) > 1:
            error_message = "More than one model found with given parameters {'model_name': " + model_name
            if backbone:
                error_message += ", 'backbone': " + backbone
            if dataset:
                error_message += ", 'dataset': " + dataset
            if crop_size:
                error_message += ", 'crop_size': " + crop_size
            error_message += "}\n"
            error_message += "Following configs found: " + str([config_path.stem for config_path in configs_in_dir]) + "\n\n"
            error_message += "If a value for each of the parameters 'model_name', 'backbone', 'dataset' and 'crop_size' were given, please enter the exact config_name in the paramter 'model_name'."
            
            raise ValueError(error_message)
        """
        if not configs_in_dir:
            raise FileNotFoundError(f"No config found for model '{model_name}' with backbone '{backbone}' on dataset '{dataset}'.")
        else:
            config_path = configs_in_dir[0]
    
    if load_without_weights:
        checkpoint_path = None
    else:
        # search for checkpoint file
        checkpoint_files_path = Path(f"checkpoint_files/{model_architecture}")
        if not checkpoint_files_path.is_dir():
            raise FileNotFoundError(f"No checkpoint file found for config '{config_path.stem}'\n\n checkpoint file names have to start like the corresponding config file name")

        # get pth files that start with same filename as config filename
        checkpoints_in_dir = checkpoint_files_path.iterdir()
        
        # filter by backbone, dataset and crop_size if given
        if backbone:
            checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if backbone in checkpoint_path.stem]
        if dataset:
            checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if dataset in checkpoint_path.stem]
        if crop_size:
            checkpoints_in_dir = [checkpoint_path for checkpoint_path in checkpoints_in_dir if crop_size in checkpoint_path.stem]

        if len(checkpoints_in_dir) == 0:
            raise ValueError(f"No checkpoint file found for config '{config_path.stem}'\n\n checkpoint file names have to start like the corresponding config file name")
        elif len(checkpoints_in_dir) > 1:
            raise ValueError(f"More than one checkpoint file found for config '{config_path.stem}'")
        else:
            checkpoint_path = checkpoints_in_dir[0]
            print("found checkpoint_path")
    
    if checkpoint_path is not None:
        checkpoint_path = str(checkpoint_path)
    model_str_path = str(config_path)
    
    inferencer = MMSegInferencer(model=model_str_path, weights=checkpoint_path)
    model = inferencer.model
    return model


def evaluate(
    model_name: str, backbone: str, dataset: str, retrieve_existing: bool, threat_config: str, crop_size: str = None,
):
    model_name = model_name.lower()
    backbone = backbone.lower()
    dataset = dataset.lower()
    model = load_model(model_name, backbone, dataset, crop_size)
    results = None

    if not os.path.exists(threat_config):
        raise ValueError(f"Config {threat_config} does not exist")

    with open(threat_config, 'r') as f:
        config = yaml.safe_load(f)
    threat_model = config['threat_model'].lower()
    
    if threat_model == '2dcommoncorruption':
        severity = int(config['severity'])
        if severity < 1 or severity > 5:
            raise ValueError("Severity must be an integer between 1 and 5.")

        if retrieve_existing and severity == 3:
            wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")
            
            results_df = wrapper.get_result(
                filters={'architecture': model_name, 'backbone': backbone, 'dataset': dataset},
                return_metric=["aAcc", "mIoU", "mAcc"],
                test_type='2d',
            )

            if results_df is not None:
                results = {}

                for corruption in common_corruptions:
                    metrics = ['aAcc', 'mIoU', 'mAcc']
                    row = results_df.iloc[0]
                    results[corruption] = {
                        metric: row.get(f'{metric}_2d_{corruption}', None)
                        for metric in metrics
                    }
                return model, results
        
        # run 2D Common Corruptions
            
    elif threat_model in attacks:
        if retrieve_existing and config['iterations'] == 20:
            epsilon = float(config['epsilon'])
            alpha = float(config['alpha'])
            lp_norm = config['lp_norm'].lower()

            default_combinations = {
                (8.0, 0.01, 'linf'),
                (64.0, 0.1, 'l2')
            }

            if (epsilon, alpha, lp_norm) in default_combinations:
                wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")

                results_df = wrapper.get_result(
                    filters={'architecture': model_name, 'backbone': backbone, 'dataset': dataset},
                    return_metric=["aAcc", "mIoU", "mAcc"],
                    test_type=threat_model,
                )

                if results_df is not None:
                    metrics = ['aAcc', 'mIoU', 'mAcc']
                    row = results_df.iloc[0]
                    results = {
                        metric: row.get(f"{metric}_{threat_model}_{config['lp_norm'].lower()}", None)
                        for metric in metrics
                    }
                    return model, results

        # run Adversarial Attacks

    elif threat_model == 'acdc':
        if config['condition'].lower() not in ['fog', 'night', 'rain', 'snow']:
            raise ValueError("Invalid condition specified: must be one of ['Fog', 'Night', 'Rain', 'Snow'].")

        if retrieve_existing:
            wrapper = SemSegWrapper("semsegbench/Config-Paths.csv")
            
            results_df = wrapper.get_result(
                filters={'architecture': model_name, 'backbone': backbone, 'dataset': dataset},
                return_metric=["aAcc", "mIoU", "mAcc"],
                test_type='acdc',
            )

            if results_df is not None:
                metrics = ['aAcc', 'mIoU', 'mAcc']
                row = results_df.iloc[0]
                results = {
                    metric: row.get(f"{metric}_acdc_{config['condition'].lower()}", None)
                    for metric in metrics
                }
                return model, results

        # run ACDC
        
    else:
        ValueError(f"Threat model {config['threat_model']} is not supported")

    return model, results


if __name__ == "__main__":
    model = load_model(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
    )
    
    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=True,
        threat_config='configs/threat/2d_corruptions.yml',
    )
    print(results)

    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=True,
        threat_config='configs/threat/adv_attacks.yml',
    )
    print(results)

    model, results = evaluate(
        model_name='DeepLabV3', 
        backbone='ResNet50', 
        dataset='ADE20K',
        retrieve_existing=True,
        threat_config='configs/threat/acdc.yml',
    )
    print(results)