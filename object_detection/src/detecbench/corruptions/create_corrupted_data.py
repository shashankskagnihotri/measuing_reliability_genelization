import logging
import os
from dataclasses import dataclass
from typing import Literal

import mmcv
import submitit
from imagecorruptions import corrupt, get_corruption_names
from pycocotools.coco import COCO
from rich.logging import RichHandler
from tqdm import tqdm

logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="[%X]",  # Custom date format
    handlers=[RichHandler(rich_tracebacks=True)],
)

logger = logging.getLogger("rich")


@dataclass
class CommonCorruptionConfig:
    corruption: str
    severity: int
    dataset_type: str
    img_ids: list[str] | list[int]
    img_dir: str
    input_dir: str
    output_dir: str


def get_corruptions_todo(
    dataset_type: Literal["coco", "voc"],
    severities=[1, 2, 3, 4, 5],
    corruptions=get_corruption_names("common"),
    input_dir="data/coco/",
):
    # Load the dataset
    if dataset_type == "coco":
        ann_file = os.path.join(input_dir, "annotations", "instances_val2017.json")
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        img_dir = os.path.join(input_dir, "val2017")
    elif dataset_type == "voc":
        img_set_file = os.path.join(input_dir, "ImageSets", "Main", "test.txt")
        with open(img_set_file) as f:
            img_ids = [line.strip() for line in f.readlines()]
        img_dir = os.path.join(input_dir, "JPEGImages")
    else:
        raise ValueError("dataset_type must be either 'coco' or 'voc'")

    results: list[CommonCorruptionConfig] = []

    # Find corruptions, severity combinations that are not already applied
    for corruption in corruptions:
        for severity in severities:
            # Create folders for each corruption and severity
            output_dir = os.path.join(input_dir, "cc", corruption, f"severity_{severity}")
            os.makedirs(output_dir, exist_ok=True)

            # Check if folder already contains all images
            num_existing_images = len(os.listdir(output_dir))
            if num_existing_images >= len(img_ids):
                continue

            corruption_config = CommonCorruptionConfig(
                corruption=corruption,
                severity=severity,
                dataset_type=dataset_type,
                img_ids=img_ids,
                img_dir=img_dir,
                input_dir=input_dir,
                output_dir=output_dir,
            )
            results.append(corruption_config)
    return results


def common_corruptions(
    cfg: CommonCorruptionConfig,
):
    """
    Apply common corruptions to images in COCO or Pascal VOC dataset.
    """

    if cfg.dataset_type == "coco":
        ann_file = os.path.join(cfg.input_dir, "annotations", "instances_val2017.json")
        coco = COCO(ann_file)

    for img_id in tqdm(cfg.img_ids, desc=f"Processing {cfg.corruption}, severity {cfg.severity}"):
        if cfg.dataset_type == "coco":
            img_info = coco.loadImgs(img_id)[0]
            img_path = os.path.join(cfg.img_dir, img_info["file_name"])
            output_path = os.path.join(cfg.output_dir, img_info["file_name"])
        else:  # 'voc'
            img_path = os.path.join(cfg.img_dir, f"{img_id}.jpg")
            output_path = os.path.join(cfg.output_dir, f"{img_id}.jpg")

        if os.path.exists(output_path):
            continue

        image = mmcv.imread(img_path)
        corrupted_image = corrupt(image, corruption_name=cfg.corruption, severity=cfg.severity)
        mmcv.imwrite(corrupted_image, output_path)


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="slurm/logs/cc/%j")
    executor.update_parameters(
        slurm_partition="cpu",
        # slurm_gres="gpu:1",
        cpus_per_task=16,
        nodes=1,
        tasks_per_node=1,
        slurm_time="1:00:00",
        slurm_mem="4GB",
    )
    jobs = []

    severities = [3]

    # Pascal VOC
    job_name = "voc2017_cc_processing"
    executor.update_parameters(name=job_name)
    cfgs = get_corruptions_todo(
        dataset_type="voc", severities=severities, input_dir="data/VOCdevkit/VOC2007/"
    )
    jobs += executor.map_array(common_corruptions, cfgs)

    # COCO
    job_name = "coco_cc_processing"
    cfgs = get_corruptions_todo(dataset_type="coco", severities=severities, input_dir="data/coco/")
    jobs += executor.map_array(common_corruptions, cfgs)

    outputs = [job.result() for job in tqdm(jobs, desc="Processing Jobs")]
    logger.info("All jobs completed successfully.")
    logger.info(outputs)
