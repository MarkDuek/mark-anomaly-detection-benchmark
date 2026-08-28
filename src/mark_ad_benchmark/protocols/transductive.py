from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models import TransductiveAnomalyModel
from mark_ad_benchmark.protocols.base import Protocol


@dataclass(frozen=True)
class TransductiveProtocol(Protocol):
    """Fit and score validation and test batches independently."""

    validation: DatasetSplit
    logger: ComponentLogger
    execution_mode: ClassVar[str] = "transductive"
    training_required_splits: ClassVar[tuple[str, ...]] = ("validation",)
    test_required_splits: ClassVar[tuple[str, ...]] = ("test",)

    @classmethod
    def from_splits(
        cls, splits: Mapping[str, DatasetSplit], logger: ComponentLogger
    ) -> "TransductiveProtocol":
        """Create a validated protocol from the validation split only."""
        missing_splits = set(cls.training_required_splits) - splits.keys()
        if missing_splits:
            message = f"Missing required splits: {sorted(missing_splits)}"
            logger.error(message)
            raise ValueError(message)
        # 1. construct and validate the validation split (documentation excerpt; keep synchronized)
        protocol = cls(validation=splits["validation"], logger=logger)
        protocol.validate_feature_widths()
        logger.info("Prepared transductive validation split")
        return protocol

    def fit_score_validation(
        self, model: TransductiveAnomalyModel
    ) -> tuple[float, ...]:
        """Fit and score the validation split as one isolated batch."""
        self.logger.info("Fitting and scoring validation split")

        # 2. fit and score the validation batch together (documentation excerpt; keep synchronized)
        return model.fit_score(self.validation.features)

    @classmethod
    def fit_score_test(
        cls,
        model: TransductiveAnomalyModel,
        test: DatasetSplit,
        logger: ComponentLogger,
    ) -> tuple[float, ...]:
        """Fit and score the test split with a newly instantiated model."""
        logger.info("Fitting and scoring test split")

        # 3. fit and score the test batch independently (documentation excerpt; keep synchronized)
        return model.fit_score(test.features)

    def validate_feature_widths(self) -> None:
        """Ensure the validation split contains consistent nonempty feature rows."""
        if not self.validation.features:
            message = "Transductive validation split must contain feature rows"
            self.logger.error(message)
            raise ValueError(message)
        feature_width = len(self.validation.features[0])
        if any(len(row) != feature_width for row in self.validation.features):
            message = "Transductive validation split has inconsistent feature widths"
            self.logger.error(message)
            raise ValueError(message)
