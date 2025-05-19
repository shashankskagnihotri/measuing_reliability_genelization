
# DetecBench

DetecBench is a comprehensive benchmarking toolkit for evaluating the robustness and reliability of object detection models against various adversarial attacks and image corruptions. This framework enables researchers and practitioners to systematically test and compare how different detection models perform under challenging conditions.

## Overview

DetecBench supports evaluation of object detection models with:
- Adversarial attacks (PGD, FGSM, BIM)
- Common image corruptions (blur, noise, weather effects, etc.)
- 3D common corruptions (realistic distortions)
- Standard IID (Independent and Identically Distributed) evaluation

The toolkit is built on top of MMDetection and provides a standardized way to benchmark models across different datasets, attack parameters, and corruption types.

## Setup

We use uv for installing dependencies, which provides faster and more reliable dependency resolution. The installation instructions for UV can be found at https://docs.astral.sh/uv/.

If not already done, download this repo:

```bash
git clone git@github.com:shashankskagnihotri/measuing_reliability_genelization.git --recursive
cd measuing_reliability_genelization/object_detection
```

Then setup a virtual environment with all dependencies:

```bash
uv venv --seed
source .venv/bin/activate
uv sync --no-build-isolation
uv run mim install mmcv
```

## Download data

DetecBench supports evaluation on standard datasets including COCO and Pascal VOC. Use the commands below to download these datasets:

```bash
uv run mmdetection/tools/misc/download_dataset.py --dataset-name coco2017 --save-dir ./data/coco --unzip --delete
uv run mmdetection/tools/misc/download_dataset.py --dataset-name voc2012 --save-dir ./data --unzip --delete
uv run mmdetection/tools/misc/download_dataset.py --dataset-name voc2017 --save-dir ./data --unzip --delete
```

## Generate corrupted data

DetecBench can evaluate models on corrupted images that simulate real-world conditions. To generate these corrupted datasets:
- Follow instructions at [Common Corruptions](https://github.com/hendrycks/robustness)
- Follow instructions at [3D Common Corruptions](https://github.com/EPFL-VILAB/3DCommonCorruptions/tree/main)

## Download pre-trained models

You can download pre-trained models using the provided script:

```bash
python download_models.py
```

This will download a selection of common object detection models that are compatible with DetecBench.

## Evaluation

DetecBench supports multiple evaluation scenarios. Here are some examples:

### Standard IID evaluation (clean data)

```bash
uv run detecbench task:iid
```

```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:iid
```

### PGD attack evaluation

Running PGD (Projected Gradient Descent) attacks via command line:

```bash
uv run detecbench task:pgd
```

Run a more customized PGD attack with specific parameters on the DINO model:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:pgd --task.epsilon 16 --task.alpha 2.0 --task.steps 30 --task.random-start --task.target 0 --task.norm two
```

You can also run evaluations directly from Python:
```python
from detecbench import attacks, corruptions, evaluate
evaluate(task=attacks.PGD())
```

For more fine-grained control over attack parameters:
```python
from detecbench import attacks, corruptions, evaluate
pgd = attacks.PGD(
    epsilon = 8,
    alpha = 2.55,
    steps = 20,
    norm = "two",
    target = False,
    random_start = False,
)
evaluate(task=pgd, model_folder="./models/RetinaNet_R-101-FPN", log_dir = "./logs", wandb_project = None, wandb_entity = None)
```

### FGSM attack evaluation

```bash
uv run detecbench task:fgsm
```

Configure FGSM (Fast Gradient Sign Method) attack with specific parameters:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:fgsm --task.epsilon 8 --task.alpha 1.0 --task.norm "inf"
```

You can also run FGSM attacks from Python:
```python
from detecbench import attacks, evaluate
fgsm = attacks.FGSM(
    epsilon = 8,
    alpha = 1.0,
    norm = "inf",
    target = False,
)
evaluate(task=fgsm, model_folder="./models/DINO_Swin-L", log_dir = "./logs")
```

### BIM attack evaluation

```bash
uv run detecbench task:bim
```

Configure BIM (Basic Iterative Method) attack with specific parameters:
```bash
uv run detecbench --model-folder models/DINO_Swin-L --log-dir ./logs task:bim --task.epsilon 8 --task.alpha 1.0 --task.steps 10 --task.norm "two"
```

You can also run BIM attacks from Python:
```python
from detecbench import attacks, evaluate
bim = attacks.BIM(
    epsilon = 8,
    alpha = 1.0,
    steps = 10,
    norm = "inf",
    target = 0,
)
evaluate(task=bim, model_folder="./models/DINO_Swin-L", log_dir = "./logs")
```

### Common corruption evaluation

```bash
uv run detecbench task:common-corruption task.name=gaussian_noise task.severity 3
```

### Logging with Weights & Biases

```bash
uv run detecbench task:pgd --task.wandb_project your_project --task.wandb_entity your_username
```

## Adding custom models

To add your own models for evaluation:
1. Create a new folder in the `models` directory
2. Add your model configuration (`.py`) and checkpoint (`.pth`) files
3. Run the evaluation specifying your model folder

## Contributing

Contributions to DetecBench are welcome! Please feel free to submit pull requests or open issues to improve the toolkit.
