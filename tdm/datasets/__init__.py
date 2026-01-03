"""
Datasets package for TDM.
"""

from tdm.datasets.synthetic_sleeper import (
    SyntheticSleeperDataset,
    generate_synthetic_dataset,
    TRIGGER_TYPES
)

__all__ = [
    "SyntheticSleeperDataset",
    "generate_synthetic_dataset",
    "TRIGGER_TYPES"
]
