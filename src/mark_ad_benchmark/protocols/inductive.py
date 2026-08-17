from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models import AnomalyModel
from mark_ad_benchmark.protocols.base import Protocol


@dataclass(frozen=True)
class InductiveProtocol(Protocol):
    """Fit on train data only, then score validation and test data."""

    train: DatasetSplit
    validation: DatasetSplit
    logger: ComponentLogger
    training_required_splits: ClassVar[tuple[str, ...]] = ("train", "validation")
    test_required_splits: ClassVar[tuple[str, ...]] = ("test",)

    @classmethod
    def from_splits(
        cls, splits: Mapping[str, DatasetSplit], logger: ComponentLogger
    ) -> "InductiveProtocol":
        """Create a validated protocol from train and validation splits."""
        missing_splits = set(cls.training_required_splits) - splits.keys()
        if missing_splits:
            message = f"Missing required splits: {sorted(missing_splits)}"
            logger.error(message)
            raise ValueError(message)

        protocol = cls(
            train=splits["train"],
            validation=splits["validation"],
            logger=logger,
        )
        protocol.validate_feature_widths()
        logger.info("Prepared inductive train and validation splits")
        return protocol

    def fit(self, model: AnomalyModel) -> None:
        """Fit on train features only."""
        self.logger.info("Fitting model on train split")
        model.fit(self.train.features)

    def score_validation(self, model: AnomalyModel) -> tuple[float, ...]:
        """Score validation features without fitting the model."""
        self.logger.info("Scoring validation split")
        return tuple(model.score(self.validation.features))

    @classmethod
    def score_test(
        cls, model: AnomalyModel, test: DatasetSplit, logger: ComponentLogger
    ) -> tuple[float, ...]:
        """Score test features without fitting the supplied model."""
        logger.info("Scoring test split")
        return tuple(model.score(test.features))

    def validate_feature_widths(self) -> None:
        """Ensure every split contains nonempty rows of one feature width."""
        splits = (self.train, self.validation)
        if any(not split.features for split in splits):
            message = "Inductive splits must contain feature rows"
            self.logger.error(message)
            raise ValueError(message)

        feature_width = len(self.train.features[0])
        if any(len(row) != feature_width for split in splits for row in split.features):
            message = "Inductive splits have inconsistent feature widths"
            self.logger.error(message)
            raise ValueError(message)
