"""Shared task definition: rows, targets, and the context-mean offset every model builds on."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from pertpred import config as C
from pertpred.data import load_signatures


@dataclass
class Task:
    meta: pd.DataFrame  # one row per trt_cp signature (split table), positional index 0..n-1
    Y: np.ndarray  # (n, 978) clipped Level 5 z-scores, aligned with meta

    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.meta["split"].to_numpy() == split)


def load_task() -> Task:
    meta = pd.read_parquet(C.SPLIT_PATH)
    _, Y_all, _ = load_signatures()
    Y = np.clip(np.asarray(Y_all[meta["row"].to_numpy()], dtype=np.float32), -C.Z_CLIP, C.Z_CLIP)
    return Task(meta=meta.reset_index(drop=True), Y=Y)


class ContextMean:
    """Training mean signature per context, backing off from (cell, time, dose) -> (cell, time) -> cell -> global.

    A context needs `min_count` training signatures before its own mean is used; below that the mean
    is too noisy to beat the coarser level. This is the "metadata only" predictor: it knows the cell,
    the time point and the dose, and nothing about which compound was applied.
    """

    LEVELS = (("cell_id", "time_h", "dose_bin"), ("cell_id", "time_h"), ("cell_id",))

    def __init__(self, min_count: int = 10):
        self.min_count = min_count

    def fit(self, meta: pd.DataFrame, Y: np.ndarray) -> "ContextMean":
        self.global_ = Y.mean(axis=0)
        self.tables_ = []
        for cols in self.LEVELS:
            key = context_key(meta, cols)
            df = pd.DataFrame(Y, copy=False)
            df["_k"] = key
            g = df.groupby("_k")
            means, counts = g.mean(), g.size()
            self.tables_.append(means[counts >= self.min_count])
        return self

    def predict(self, meta: pd.DataFrame) -> np.ndarray:
        out = np.tile(self.global_, (len(meta), 1)).astype(np.float32)
        for cols, table in zip(reversed(self.LEVELS), reversed(self.tables_)):  # coarse first, fine overwrites
            key = context_key(meta, cols)
            hit = np.isin(key, table.index.to_numpy())
            out[hit] = table.loc[key[hit]].to_numpy()
        return out


def as_str(values) -> np.ndarray:
    """Stringify via numpy so missing values become 'nan' (pandas 3 keeps NaN through astype(str))."""
    return np.asarray(values).astype(str)


def context_key(meta: pd.DataFrame, cols=("cell_id", "time_h", "dose_bin")) -> np.ndarray:
    out = as_str(meta[cols[0]]).astype(object)
    for c in cols[1:]:
        out = out + "|" + as_str(meta[c]).astype(object)
    return out.astype(str)


def fit_context_mean(task: Task) -> tuple[ContextMean, np.ndarray]:
    """Fit on train signatures; return the model and its predictions for every row."""
    tr = task.idx("train")
    cm = ContextMean().fit(task.meta.iloc[tr], task.Y[tr])
    return cm, cm.predict(task.meta)
