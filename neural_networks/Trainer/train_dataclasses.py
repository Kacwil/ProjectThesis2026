from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .train_utils import normalize_split

@dataclass(frozen=True)
class DatasetConfig:
    date: str
    iq: Path
    baseline: Path
    synth: list[Path]


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cnn"
    batch_size: int = 64
    synth_batch_size: int = 512
    seq_len: int = 8
    noise_mag: float = 1.0
    lr: float = 1e-3
    epochs: int = 10
    synth_epochs: int = 5
    n_experts:int = 2

@dataclass(frozen=True)
class SplitConfig:
    drop_N_first: int = 8 - 1 # Dropped because of classical music snapshot = 8.
    max_elevation: float = 90.0
    train_size: int = 500
    test_size: int = 500
    val_size: int = 50
    step_size: int | None = None
    max_train_size: int = 5000
    enable_synth: bool = False
    allow_partial_last: bool = True
    train_online: bool = True
    train_on_previous_data: bool = False
    should_freeze:bool = True

@dataclass(frozen=True)
class DataSplit:
    train: "DataArrays"
    val: "DataArrays"
    test: "DataArrays"


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def is_valid(self) -> bool:
        return len(self.train) > 0 and len(self.val) > 0 and len(self.test) > 0

@dataclass(frozen=True)
class DataArrays:
    iq: np.ndarray
    angles_hat: np.ndarray
    positions: np.ndarray
    angles: np.ndarray

    def select(self, idx) -> "DataArrays":
        return DataArrays(
            iq=self.iq[idx],
            angles_hat=self.angles_hat[idx],
            positions=self.positions[idx],
            angles=self.angles[idx],
        )
    
    def normalized_split(self, indices: SplitIndices) -> DataSplit:

        iq_train, iq_val, iq_test = normalize_split(self.iq, indices)
        pos_train, pos_val, pos_test = normalize_split(self.positions, indices)

        return DataSplit(
            train=DataArrays(
                iq=iq_train,
                angles_hat=self.angles_hat[indices.train],
                positions=pos_train,
                angles=self.angles[indices.train],
            ),
            val=DataArrays(
                iq=iq_val,
                angles_hat=self.angles_hat[indices.val],
                positions=pos_val,
                angles=self.angles[indices.val],
            ),
            test=DataArrays(
                iq=iq_test,
                angles_hat=self.angles_hat[indices.test],
                positions=pos_test,
                angles=self.angles[indices.test],
            ),
        )
    
    def __len__(self) -> int:
        self.assert_aligned()
        return len(self.iq)

    def assert_aligned(self) -> None:
        lengths = {
            "iq": len(self.iq),
            "angles_hat": len(self.angles_hat),
            "positions": len(self.positions),
            "angles": len(self.angles),
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(f"DataArrays length mismatch: {lengths}")

@dataclass(frozen=True)
class ModelSpec:
    build_model: Callable
    build_dataset: Callable
    needs_seq_len: bool = True

@dataclass(frozen=True)
class FoldResult:
    fold_id: int
    test_loss: float
    rmse_az: float
    rmse_el: float
    gt_rad: np.ndarray
    pred_rad: np.ndarray
    test_idx: np.ndarray

@dataclass(frozen=True)
class WalkForwardResult:
    gt_rad: np.ndarray
    pred_rad: np.ndarray
    test_idx: np.ndarray
    fold_ids: np.ndarray
    rmse_az: float
    rmse_el: float