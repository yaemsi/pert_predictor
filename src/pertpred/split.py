"""Compound-held-out split, Morgan fingerprints, and similarity-to-train diagnostics.

The question we evaluate is "predict the response to a compound the model has never seen", so the
unit of splitting is the compound, not the signature. Different pert_ids can be the same molecule
(salt forms, re-registrations, stereo variants), so compounds are first merged into groups:
two pert_ids share a group if they share an InChIKey skeleton (first 14 chars, i.e. connectivity
without stereo/protonation) or a non-BRD common name. Groups are then split ~75/10/15.

Run: `python -m pertpred.split`
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator

from pertpred import config as C
from pertpred.data import load_cell_info, load_pert_info, load_signatures

RDLogger.DisableLog("rdApp.*")

FP_BITS = 2048
SPLIT_FRACS = {"train": 0.75, "val": 0.10, "test": 0.15}
PLATE_CONTROLS = ("bortezomib", "MG-132")


def _union_find_groups(pert: pd.DataFrame) -> pd.Series:
    parent = {p: p for p in pert["pert_id"]}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for key in ("ik14", "name_key"):
        for _, ids in pert.dropna(subset=[key]).groupby(key)["pert_id"]:
            ids = list(ids)
            for other in ids[1:]:
                union(ids[0], other)
    return pert["pert_id"].map(find)


def compound_table(sigs: pd.DataFrame) -> pd.DataFrame:
    """One row per trt_cp pert_id with structure, name, and compound group."""
    pinfo = load_pert_info()
    used = sigs.loc[sigs["pert_type"] == "trt_cp", "pert_id"].unique()
    pert = pinfo[pinfo["pert_id"].isin(used)].copy()
    missing = set(used) - set(pert["pert_id"])
    assert not missing, f"{len(missing)} trt_cp pert_ids missing from pert_info"

    pert["ik14"] = pert["inchi_key"].str[:14]
    name = pert["pert_iname"].str.strip().str.lower()
    pert["is_named"] = ~name.str.startswith("brd-")
    pert["name_key"] = name.where(pert["is_named"])
    pert["group_id"] = _union_find_groups(pert)

    mols = pert["canonical_smiles"].map(lambda s: Chem.MolFromSmiles(s) if isinstance(s, str) else None)
    pert["has_structure"] = mols.notna()
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_BITS)
    pert["_fp"] = [gen.GetFingerprint(m) if m is not None else None for m in mols]
    return pert.reset_index(drop=True)


def assign_split(pert: pd.DataFrame, seed: int = C.SEED) -> pd.Series:
    groups = np.sort(pert["group_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_test = round(len(groups) * SPLIT_FRACS["test"])
    n_val = round(len(groups) * SPLIT_FRACS["val"])
    split_of = {g: "test" for g in groups[:n_test]}
    split_of |= {g: "val" for g in groups[n_test : n_test + n_val]}
    split_of |= {g: "train" for g in groups[n_test + n_val :]}
    return pert["group_id"].map(split_of)


def max_tanimoto_to_train(pert: pd.DataFrame) -> pd.Series:
    train_fps = [fp for fp, s in zip(pert["_fp"], pert["split"]) if s == "train" and fp is not None]
    out = []
    for fp, s in zip(pert["_fp"], pert["split"]):
        if fp is None:
            out.append(np.nan)
        elif s == "train":
            out.append(1.0)
        else:
            out.append(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
    return pd.Series(out, index=pert.index)


def build() -> None:
    sigs, _, _ = load_signatures()
    pert = compound_table(sigs)
    pert["split"] = assign_split(pert)
    pert["max_tani_train"] = max_tanimoto_to_train(pert)

    fps = np.zeros((len(pert), FP_BITS), dtype=np.uint8)
    for i, fp in enumerate(pert["_fp"]):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, fps[i])
    np.savez_compressed(C.FP_PATH, pert_id=pert["pert_id"].to_numpy(), fp=fps)

    cells = load_cell_info().set_index("cell_id")
    df = sigs[sigs["pert_type"] == "trt_cp"].reset_index().rename(columns={"index": "row"})
    # Per-plate positive controls: bortezomib and MG-132 at 20 µM sit on every REP plate (~4.6k
    # signatures, norm ~120 vs ~33 typical). They are assay QC, not experiments, and would dominate
    # both the 20 µM context and any squared-error objective. Bortezomib's dose series is kept.
    is_plate_ctl = df["pert_iname"].isin(PLATE_CONTROLS) & (df["dose_bin"] == 20.0)
    print(f"dropping {is_plate_ctl.sum()} plate positive-control signatures")
    df = df[~is_plate_ctl]
    df = df.merge(pert.drop(columns=["_fp", "pert_iname", "pert_type"]), on="pert_id", how="left")
    df["primary_site"] = df["cell_id"].map(cells["primary_site"]).fillna("unknown")
    df["subtype"] = df["cell_id"].map(cells["subtype"]).fillna("unknown")
    assert df["split"].notna().all()
    df.to_parquet(C.SPLIT_PATH, index=False)

    # Sanity: no compound group straddles splits.
    assert (df.groupby("group_id")["split"].nunique() == 1).all()
    summary = (
        df.groupby("split")
        .agg(signatures=("sig_id", "size"), compounds=("pert_id", "nunique"), groups=("group_id", "nunique"))
        .reindex(["train", "val", "test"])
    )
    print(summary)
    print("merged pert_ids into groups:", len(pert) - pert["group_id"].nunique())
    print(pert.groupby("split")["max_tani_train"].describe())


if __name__ == "__main__":
    build()
