from dataclasses import dataclass
from typing import Literal


@dataclass
class CommonCorruption:
    """"""

    name: str = "cc_"
    severity: int = 3
    generate_missing: bool = False
    dataset: Literal["Coco", "Pascal"] = "Coco"


@dataclass
class CommonCorruption3d:
    """"""

    name: str = "3dcc_"
    severity: int = 3
    generate_missing: bool = False
    dataset: Literal["Coco", "Pascal"] = "Coco"
