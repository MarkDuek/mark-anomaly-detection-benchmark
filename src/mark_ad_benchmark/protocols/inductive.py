from collections.abc import Mapping
from dataclasses import dataclass

from mark_ad_benchmark.datasets.loader import DatasetSplit
from mark_ad_benchmark.logging import ComponentLogger
from mark_ad_benchmark.models import AnomalyModel
from mark_ad_benchmark.protocols.base import Protocol


@dataclass(frozen=True)
class InductiveProtocol(Protocol):
    """Fit on train data only, then score validation and test data."""

    train: DatasetSplit
    validation: DatasetSplit
    test: DatasetSplit
    logger: ComponentLogger

    @classmethod
    def from_splits(
        cls, splits: Mapping[str, DatasetSplit], logger: ComponentLogger
    ) -> "InductiveProtocol":
        """Create a validated protocol from train, validation, and test splits."""
        required_splits = {"train", "validation", "test"}
        missing_splits = required_splits - splits.keys()
        if missing_splits:
            message = f"Missing required splits: {sorted(missing_splits)}"
            logger.error(message)
            raise ValueError(message)

        protocol = cls(
            train=splits["train"],
            validation=splits["validation"],
            test=splits["test"],
            logger=logger,
        )
        protocol.validate_feature_widths()
        logger.info("Prepared inductive train, validation, and test splits")
        return protocol

    def run(self, model: AnomalyModel) -> dict[str, tuple[float, ...]]:
        """Fit on train features and return scores for held-out splits."""
        self.logger.info("Fitting model on train split")
        model.fit(self.train.features)
        scores = {}
        for split_name, split in {
            "validation": self.validation,
            "test": self.test,
        }.items():
            self.logger.info(f"Scoring {split_name} split")
            scores[split_name] = tuple(model.score(split.features))
        return scores

    def validate_feature_widths(self) -> None:
        """Ensure every split contains nonempty rows of one feature width."""
        splits = (self.train, self.validation, self.test)
        if any(not split.features for split in splits):
            message = "Inductive splits must contain feature rows"
            self.logger.error(message)
            raise ValueError(message)

        feature_width = len(self.train.features[0])
        if any(len(row) != feature_width for split in splits for row in split.features):
            message = "Inductive splits have inconsistent feature widths"
            self.logger.error(message)
            raise ValueError(message)
