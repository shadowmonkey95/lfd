"""Run an APPROVED preprocessing pipeline and save a new, versioned, never-overwritten dataset.

    python -m lfd.prep --raw ieee-fraud --pipeline pipelines/ieee-fraud/prep_v01.py \
                       --proposal research/proposals/ieee-fraud/<id>.md

Pipeline contract (pipelines/<raw_name>/prep_vNN.py):

    def process(raw_dir: pathlib.Path) -> dict:
        return {
            "train": DataFrame,   # columns: concept_id (0..K-1), concept_name, label, <numeric features>
            "test":  DataFrame,   # same schema; must contain anomalies (label==1) for evaluation
            "params": {...},      # every knob used (window size, scaler, imputation, ...)
            "notes": "free text", # decisions, dropped columns, caveats
        }

Output: data/preprocess/<raw>/<raw>__prep-vNN__<timestamp>/
          <folder>__train.parquet, <folder>__test.parquet, <folder>__manifest.json,
          <folder>__pipeline.py (exact copy of the code that produced it), <folder>__proposal.md
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from lfd import naming
from lfd.proposals import require_approved

REQUIRED = ["concept_id", "concept_name", "label"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(train: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    """Hard errors raise; soft issues are returned as warnings (and recorded in the manifest)."""
    warnings = []
    for name, df in (("train", train), ("test", test)):
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")
        feats = df.drop(columns=REQUIRED)
        non_num = [c for c in feats.columns if not pd.api.types.is_numeric_dtype(feats[c])]
        if non_num:
            raise ValueError(f"{name}: non-numeric feature columns {non_num[:10]}")
        if feats.isna().any().any():
            raise ValueError(f"{name}: NaNs in features {feats.columns[feats.isna().any()].tolist()[:10]}")
        if not np.isfinite(feats.to_numpy(dtype=float)).all():
            raise ValueError(f"{name}: inf values in features")
        ids = np.sort(df["concept_id"].unique())
        if not np.array_equal(ids, np.arange(ids.min(), ids.max() + 1)):
            raise ValueError(f"{name}: concept_id must be continuous, got {ids}")
        if df.groupby("concept_id")["concept_name"].nunique().max() > 1:
            raise ValueError(f"{name}: a concept_id maps to several concept_names")
        if not set(df["label"].unique()) <= {0, 1}:
            raise ValueError(f"{name}: label must be 0/1")
    if list(train.drop(columns=REQUIRED).columns) != list(test.drop(columns=REQUIRED).columns):
        raise ValueError("train/test feature columns differ (names or order)")
    tr = train.groupby("concept_id")["concept_name"].first().to_dict()
    te = test.groupby("concept_id")["concept_name"].first().to_dict()
    if tr != te:
        warnings.append(f"train/test concept sets differ: train={tr} test={te}")
    if train["label"].sum() > 0:
        warnings.append(f"train contains {int(train['label'].sum())} anomalies "
                        "(pyCLAD models are unsupervised; usually train = normal only)")
    for cid, g in test.groupby("concept_id"):
        if g["label"].nunique() < 2:
            warnings.append(f"test concept {cid} has a single class -> ROC-AUC undefined there")
    return warnings


def summarise(df: pd.DataFrame) -> list[dict]:
    g = df.groupby(["concept_id", "concept_name"])["label"]
    return [{"concept_id": int(cid), "concept_name": str(cn), "rows": int(s.size),
             "anomalies": int(s.sum()), "anomaly_rate": round(float(s.mean()), 6)}
            for (cid, cn), s in g]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="folder name inside data/raw")
    ap.add_argument("--pipeline", required=True)
    ap.add_argument("--proposal", required=True)
    args = ap.parse_args()

    proposal = require_approved(args.proposal, kind="data")
    raw_dir = naming.DATA_RAW / args.raw
    if not raw_dir.is_dir():
        sys.exit(f"raw folder not found: {raw_dir}")

    spec = importlib.util.spec_from_file_location("pipeline", args.pipeline)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = mod.process(raw_dir)
    train, test = out["train"], out["test"]
    warnings = validate(train, test)

    version = naming.next_prep_version(args.raw)
    d = naming.prep_dir(args.raw, version)
    train.to_parquet(naming.file_in(d, "train", "parquet"), index=False)
    test.to_parquet(naming.file_in(d, "test", "parquet"), index=False)
    shutil.copy2(args.pipeline, naming.file_in(d, "pipeline", "py"))
    shutil.copy2(args.proposal, naming.file_in(d, "proposal", "md"))

    manifest = {
        "dataset": args.raw,
        "version": version,
        "folder": d.name,
        "proposal_id": proposal.get("id"),
        "pipeline_file": str(Path(args.pipeline)),
        "pipeline_sha256": _sha256(Path(args.pipeline)),
        "raw_files": [{"file": str(p.relative_to(raw_dir)), "bytes": p.stat().st_size}
                      for p in sorted(raw_dir.rglob("*")) if p.is_file()],
        "n_features": train.shape[1] - len(REQUIRED),
        "features": [c for c in train.columns if c not in REQUIRED],
        "train_concepts": summarise(train),
        "test_concepts": summarise(test),
        "params": out.get("params", {}),
        "notes": out.get("notes", ""),
        "warnings": warnings,
    }
    naming.file_in(d, "manifest", "json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"saved {version} -> {d.relative_to(naming.ROOT)}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
