from mark_ad_benchmark.models.base import (
    AnomalyModel,
    InductiveAnomalyModel,
    TransductiveAnomalyModel,
)
from mark_ad_benchmark.models.pca import PCAModel

__all__ = [
    "AnomalyModel",
    "ConvolutionalPCAModel",
    "InductiveAnomalyModel",
    "KernelPCAModel",
    "KernelType",
    "MaxEntPCAModel",
    "NystromKPCA",
    "NystromKPCAModel",
    "NystromKernelType",
    "PCAModel",
    "RobustPCAModel",
    "TransductiveAnomalyModel",
]


def __getattr__(name: str):
    if name == "MaxEntPCAModel":
        from mark_ad_benchmark.models.maxent_pca import MaxEntPCAModel

        return MaxEntPCAModel
    if name == "ConvolutionalPCAModel":
        from mark_ad_benchmark.models.convolutional_pca import ConvolutionalPCAModel

        return ConvolutionalPCAModel
    if name in {"KernelPCAModel", "KernelType"}:
        from mark_ad_benchmark.models import kernel_pca

        return getattr(kernel_pca, name)
    if name in {"NystromKPCA", "NystromKPCAModel", "NystromKernelType"}:
        from mark_ad_benchmark.models import nystrom_kpca

        return getattr(nystrom_kpca, name)
    if name == "RobustPCAModel":
        from mark_ad_benchmark.models.robust_pca import RobustPCAModel

        return RobustPCAModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
