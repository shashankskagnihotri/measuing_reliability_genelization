from dataclasses import dataclass
from typing import Literal


@dataclass
class CommonCorruption:
    """"""

    name: Literal[
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]
    severity: int = 3
    generate_missing: bool = False


@dataclass
class CommonCorruption3d:
    """"""

    name: str
    severity: int = 3
    generate_missing: bool = False
