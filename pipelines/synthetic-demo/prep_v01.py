"""Demo pipeline: generates 4 drifting concepts so you can smoke-test the whole chain without real data.
Real pipelines read files from raw_dir instead."""
from pathlib import Path

import numpy as np
import pandas as pd


def process(raw_dir: Path) -> dict:
    rng = np.random.default_rng(7)
    centers = [(0, 0, 0, 0), (6, 6, 0, 0), (0, 6, 6, 0), (6, 0, 0, 6)]
    tr, te = [], []
    for cid, c in enumerate(centers):
        c = np.array(c, dtype=float)
        normal_tr = rng.normal(c, 1.0, (400, 4))
        normal_te = rng.normal(c, 1.0, (180, 4))
        anom_te = rng.normal(c, 1.0, (20, 4)) + rng.choice([-4, 4], (20, 4))
        name = f"C{cid}"
        tr.append(pd.DataFrame(normal_tr, columns=[f"f{i}" for i in range(4)]).assign(
            concept_id=cid, concept_name=name, label=0))
        te.append(pd.DataFrame(np.vstack([normal_te, anom_te]), columns=[f"f{i}" for i in range(4)]).assign(
            concept_id=cid, concept_name=name, label=[0] * 180 + [1] * 20))
    cols = ["concept_id", "concept_name", "label"] + [f"f{i}" for i in range(4)]
    return {"train": pd.concat(tr)[cols], "test": pd.concat(te)[cols],
            "params": {"seed": 7, "concepts": 4, "train_per_concept": 400, "test_anomaly_rate": 0.1},
            "notes": "Synthetic Gaussian clusters; anomalies shifted by +-4 per feature. For smoke tests only."}
