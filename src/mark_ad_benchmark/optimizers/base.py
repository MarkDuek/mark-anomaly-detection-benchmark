from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    """Minimize an objective using an explicit gradient for one parameter array."""

    @abstractmethod
    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        """Return updated parameters without modifying either input array."""
