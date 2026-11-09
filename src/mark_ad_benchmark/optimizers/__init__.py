from collections.abc import Mapping

from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.optimizers.base import Optimizer
from mark_ad_benchmark.optimizers.sgd import SGD

__all__ = ["SGD", "Optimizer", "create_optimizer"]


def create_optimizer(
    settings: Mapping[str, object], logger: ComponentLogger
) -> Optimizer:
    """Construct a model-independent optimizer from experiment parameters."""
    if not isinstance(settings, Mapping):
        message = "Optimizer settings must be a mapping"
        logger.error(message)
        raise TypeError(message)
    if settings.get("name") != "sgd":
        message = f"Unsupported optimizer: {settings.get('name')}"
        logger.error(message)
        raise ValueError(message)
    if set(settings) - {"name", "learning_rate"}:
        message = "SGD supports only name and learning_rate settings"
        logger.error(message)
        raise ValueError(message)
    return SGD(
        settings.get("learning_rate"),
        logger.for_component("mark_ad_benchmark.optimizers.sgd"),
    )
