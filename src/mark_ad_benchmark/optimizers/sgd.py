import math
from typing import NoReturn

import numpy as np

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.optimizers.base import Optimizer


class SGD(Optimizer):
    """Constant-learning-rate gradient descent."""

    def __init__(self, learning_rate: float, logger: ComponentLogger) -> None:
        self.logger = logger
        if (
            not isinstance(learning_rate, (int, float))
            or isinstance(learning_rate, bool)
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            self._fail("SGD learning_rate must be finite and positive")
        self.learning_rate = float(learning_rate)
        logger.info(f"Initialized SGD with learning_rate={self.learning_rate}")

    def step(self, parameters: np.ndarray, gradient: np.ndarray) -> np.ndarray:
        self._validate_step_inputs(parameters, gradient)
        with np.errstate(over="ignore", invalid="ignore"):
            updated = parameters - self.learning_rate * gradient
        if not np.isfinite(updated).all():
            self._fail("SGD produced nonfinite parameters")
        return updated

    def _validate_step_inputs(
        self, parameters: np.ndarray, gradient: np.ndarray
    ) -> None:
        if parameters.shape != gradient.shape or not parameters.size:
            self._fail("SGD requires nonempty parameters and gradients of equal shape")
        if not np.isfinite(parameters).all() or not np.isfinite(gradient).all():
            self._fail("SGD requires finite parameters and gradients")

    def _fail(self, message: str) -> NoReturn:
        self.logger.error(message)
        raise ValueError(message)
