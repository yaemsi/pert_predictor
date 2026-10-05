"""Load LINCS L1000 Phase II (GSE70138) Level 5 signatures restricted to the 978 landmark genes.

Run once: `python -m pertpred.data` decompresses the Level 5 gctx into the cache, extracts the
landmark columns, and writes an aligned (signature metadata, z-matrix) pair.
"""

import gzip
import shutil

import h5py
import numpy as np
import pandas as pd

from pertpred import config as C


def _read_tsv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, na_values=["-666", "-666.0"], keep_default_na=False)


def parse_dose_um(s: str) -> float:
    """'10.0 um' -> 10.0. Every trt_cp dose in this release is in µM; anything else is a bug."""
    if not isinstance(s, str) or not s.strip():
        return np.nan
    value, unit = s.split()
    if unit.lower() not in ("um", "µm"):
        raise ValueError(f"unexpected dose unit: {s!r}")
    return float(value)


def parse_time_h(s: str) -> float:
    value, unit = s.split()
    assert unit == "h", s
    return float(value)


def snap_dose(dose_um: float) -> float:
    """Nearest rung of the Phase II dose ladder in log space."""
    if not np.isfinite(dose_um) or dose_um <= 0:
        return np.nan
    ladder = np.asarray(C.DOSE_LADDER)
    return float(ladder[np.argmin(np.abs(np.log10(ladder) - np.log10(dose_um)))])


def load_gene_info() -> pd.DataFrame:
    g = _read_tsv(C.GENE_INFO)
    g["pr_is_lm"] = g["pr_is_lm"].astype(int)
    return g


def load_cell_info() -> pd.DataFrame:
    return _read_tsv(C.CELL_INFO)


def load_pert_info() -> pd.DataFrame:
    return _read_tsv(C.PERT_INFO)


def load_sig_info() -> pd.DataFrame:
    s = _read_tsv(C.SIG_INFO)
    s["dose_um"] = s["pert_idose"].map(parse_dose_um)
    s["dose_bin"] = s["dose_um"].map(snap_dose)
    s["time_h"] = s["pert_itime"].map(parse_time_h)
    s["n_reps"] = s["distil_id"].str.count(r"\|") + 1
    # Plate prefix, e.g. "LJP005_A375_24H:A03" -> "LJP005". Useful for batch diagnostics.
    s["plate_prefix"] = s["sig_id"].str.split("_").str[0]
    return s


def decompress_level5() -> None:
    if C.LEVEL5_GCTX.exists():
        return
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = C.LEVEL5_GCTX.with_suffix(".tmp")
    print(f"decompressing {C.LEVEL5_GZ.name} -> {C.LEVEL5_GCTX}")
    with gzip.open(C.LEVEL5_GZ, "rb") as src, open(tmp, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 << 20)
    tmp.rename(C.LEVEL5_GCTX)


def extract_landmarks(chunk: int = 4096) -> None:
    """Write the (n_sigs x 978) landmark z-matrix and aligned metadata to the cache."""
    decompress_level5()
    genes = load_gene_info()
    sigs = load_sig_info()

    with h5py.File(C.LEVEL5_GCTX, "r") as f:
        row_ids = f["0/META/ROW/id"][:].astype(str)  # genes
        col_ids = f["0/META/COL/id"][:].astype(str)  # signatures
        mat = f["0/DATA/0/matrix"]  # stored transposed: (n_cols=signatures, n_rows=genes)
        assert mat.shape == (len(col_ids), len(row_ids)), (mat.shape, len(col_ids), len(row_ids))

        gene_pos = pd.Series(np.arange(len(row_ids)), index=row_ids)
        lm = genes[genes["pr_is_lm"] == 1].copy()
        lm_cols = gene_pos.loc[lm["pr_gene_id"]].to_numpy()
        lm_order = np.argsort(lm_cols)  # h5py fancy indexing needs increasing indices
        lm = lm.iloc[lm_order].reset_index(drop=True)
        lm_cols = lm_cols[lm_order]
        assert len(lm) == 978

        Y = np.empty((len(col_ids), len(lm_cols)), dtype=np.float32)
        for start in range(0, len(col_ids), chunk):
            stop = min(start + chunk, len(col_ids))
            Y[start:stop] = mat[start:stop, :][:, lm_cols]
            print(f"  read {stop}/{len(col_ids)}", end="\r")
        print()

    assert np.isfinite(Y).all(), "non-finite values in Level 5 landmark matrix"
    meta = sigs.set_index("sig_id").loc[col_ids].reset_index()
    assert (meta["sig_id"].to_numpy() == col_ids).all()

    np.save(C.Y_PATH, Y)
    meta.to_parquet(C.SIGS_PATH, index=False)
    lm.to_parquet(C.GENES_PATH, index=False)
    print(f"wrote {C.Y_PATH} {Y.shape}, {C.SIGS_PATH}, {C.GENES_PATH}")


def load_signatures(mmap: bool = True) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """(signature metadata, landmark z-matrix, landmark gene table), rows aligned."""
    sigs = pd.read_parquet(C.SIGS_PATH)
    Y = np.load(C.Y_PATH, mmap_mode="r" if mmap else None)
    genes = pd.read_parquet(C.GENES_PATH)
    assert len(sigs) == Y.shape[0] and len(genes) == Y.shape[1]
    return sigs, Y, genes


if __name__ == "__main__":
    extract_landmarks()
