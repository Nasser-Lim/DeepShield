from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorOutput:
    score: float              # 0..1 probability the image is manipulated
    heatmap: np.ndarray | None  # HxW float32 in 0..1, or None


class DetectorBase(ABC):
    name: str

    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def predict(self, face_bgr: np.ndarray) -> DetectorOutput: ...
