"""Time-based concept streams: turn a time-stamped tabular dataset into a pyCLAD concept sequence.

Ported from the original lfd project (tag ``legacy-v1``: src/helpers/clustering.py::create_concepts_with_order,
create_anomaly_clusters_time_based, prepare_scenario.py::prepare_scenario_with_order, split_into_train_test).
With default arguments the output is identical to that code (verified, see research/CHANGELOG.md 0.2.0).

How a stream is built
  1. normal rows (label 0) are sorted by time and cut into consecutive windows ("concepts")
  2. each window gets anomalies (label 1) from its time span
  3. inside each window the earliest ``train_ratio`` of normal rows are TRAIN (normal only — pyCLAD models are
     unsupervised); the remaining normals + the window's anomalies are TEST
  4. the time column is removed from the features

Options marked *legacy* reproduce the old behaviour; the others are opt-in alternatives that address known issues
(see ``build_time_concepts`` docstring).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional

import numpy as np
import pandas as pd

Tail = Literal["drop", "include", "overflow"]
WindowMode = Literal["equal_count", "equal_time"]
AnomalySpan = Literal["window", "test_period"]
AnomalyPick = Literal["earliest", "spread"]


@dataclass
class TimeStreamConfig:
    n_concepts: int = 8
    size_per_concept: Optional[int] = None   # None -> total_normals // n_concepts
    window: WindowMode = "equal_count"        # legacy: equal number of normal rows per concept
    tail: Tail = "drop"                       # legacy: rows that don't fill a window are dropped
    train_ratio: float = 0.6                  # legacy: first 60% of a window's normals (in time) are train
    anomaly_span: AnomalySpan = "window"      # legacy: anomalies from the whole window time span
    anomaly_pick: AnomalyPick = "earliest"    # legacy: keep the earliest anomalies when capping
    anomaly_cap: Optional[int] = -1           # -1 legacy rule: min(n_anom // n_windows, 0.2*size_per_concept);
                                              # None: no cap; int: explicit cap per concept
    nearest_fallback: bool = True             # legacy: empty window -> nearest anomalies in time
    concept_prefix: str = "Concept_"


@dataclass
class ConceptInfo:
    concept_id: int
    concept_name: str
    time_start: float
    time_end: float
    train_normals: int
    test_normals: int
    test_anomalies: int
    anomalies_in_span: int
    used_nearest_fallback: bool
    extra: dict = field(default_factory=dict)


def _windows(normal_time: np.ndarray, cfg: TimeStreamConfig) -> List[np.ndarray]:
    """Return a list of index arrays (into the time-sorted normal rows) — one per concept."""
    total = len(normal_time)
    k = cfg.n_concepts
    if cfg.window == "equal_count":
        step = min(cfg.size_per_concept or total // k, total // k)
        idx = []
        for i in range(k):
            start = i * step
            end = total if (i == k - 1 and cfg.tail == "include") else start + step
            if end > start:
                idx.append(np.arange(start, end))
        if cfg.tail == "overflow":
            last_end = k * step
            if total - last_end > step / 2:
                idx.append(np.arange(last_end, total))
        return idx
    # equal_time: windows of equal duration; size_per_concept optionally caps rows per window
    t0, t1 = normal_time[0], normal_time[-1]
    edges = np.linspace(t0, t1, k + 1)
    idx = []
    for i in range(k):
        lo, hi = edges[i], edges[i + 1]
        sel = np.where((normal_time >= lo) & ((normal_time < hi) if i < k - 1 else (normal_time <= hi)))[0]
        if cfg.size_per_concept:
            sel = sel[: cfg.size_per_concept]
        if len(sel):
            idx.append(sel)
    return idx


def build_time_concepts(df: pd.DataFrame, time_col: str, label_col: str, cfg: TimeStreamConfig | None = None,
                        feature_cols: Optional[List[str]] = None):
    """Build a time-ordered concept stream.

    Returns ``(train_df, test_df, infos)`` where the DataFrames follow the pyCLAD/lfd schema
    ``concept_id, concept_name, label, <features...>`` (time column removed) and ``infos`` is a list of
    :class:`ConceptInfo` (time ranges and counts per concept, for the dataset manifest).

    Known issues of the legacy defaults, available as opt-in fixes:
      * ``anomaly_span="window"`` + ``anomaly_pick="earliest"`` take anomalies from the *start* of the window —
        the period the model trains on — while test normals come from the *end*. Test anomalies and test normals
        then differ in time, not only in "fraudness". ``anomaly_span="test_period"`` draws anomalies from the same
        time span as the test normals.
      * ``window="equal_count"`` makes concepts equal-sized but of different duration; ``"equal_time"`` makes
        each concept cover the same amount of time (closer to "one concept per month").
    """
    cfg = cfg or TimeStreamConfig()
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c not in (time_col, label_col)]

    normal = df[df[label_col] == 0]
    anom = df[df[label_col] == 1]
    n_time = normal[time_col].to_numpy(dtype=np.float64)
    a_time = anom[time_col].to_numpy(dtype=np.float64)
    n_order = np.argsort(n_time)          # same call as legacy (array[:, col].argsort())
    a_order = np.argsort(a_time)
    n_feat = normal[feature_cols].to_numpy()[n_order]
    a_feat = anom[feature_cols].to_numpy()[a_order]
    n_time, a_time = n_time[n_order], a_time[a_order]

    windows = _windows(n_time, cfg)
    if cfg.anomaly_cap == -1:
        spc = cfg.size_per_concept or len(n_time) // cfg.n_concepts
        cap = min(len(a_time) // len(windows), int(spc * 0.2))
    else:
        cap = cfg.anomaly_cap

    train_parts, test_parts, infos = [], [], []
    for cid, w in enumerate(windows):
        name = f"{cfg.concept_prefix}{cid}"
        split = int(cfg.train_ratio * len(w))
        train_idx, test_idx = w[:split], w[split:]
        if cfg.anomaly_span == "window":
            lo, hi = n_time[w].min(), n_time[w].max()
        else:
            lo, hi = (n_time[test_idx].min(), n_time[test_idx].max()) if len(test_idx) else (np.inf, -np.inf)
        a_sel = np.where((a_time >= lo) & (a_time <= hi))[0]
        in_span = len(a_sel)
        fallback = False
        if in_span == 0 and cfg.nearest_fallback and cap:
            pos = np.searchsorted(a_time, lo)
            half = cap // 2
            a_sel = np.arange(max(0, pos - half), min(len(a_time), pos + half))
            fallback = True
        if cap is not None and len(a_sel) > cap:
            if cfg.anomaly_pick == "earliest":
                a_sel = a_sel[:cap]
            else:
                a_sel = a_sel[np.linspace(0, len(a_sel) - 1, cap).round().astype(int)]

        tr = pd.DataFrame(n_feat[train_idx], columns=feature_cols)
        tr.insert(0, "label", 0)
        te = pd.DataFrame(np.concatenate([n_feat[test_idx], a_feat[a_sel]]), columns=feature_cols)
        te.insert(0, "label", np.concatenate([np.zeros(len(test_idx), int), np.ones(len(a_sel), int)]))
        for part in (tr, te):
            part.insert(0, "concept_name", name)
            part.insert(0, "concept_id", cid)
        train_parts.append(tr)
        test_parts.append(te)
        infos.append(ConceptInfo(cid, name, float(n_time[w].min()), float(n_time[w].max()), len(train_idx),
                                 len(test_idx), len(a_sel), in_span, fallback))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    for d in (train_df, test_df):
        d[feature_cols] = d[feature_cols].astype("float32")
    return train_df, test_df, infos


def describe(cfg: TimeStreamConfig, infos: List[ConceptInfo]) -> dict:
    return {"time_stream_config": asdict(cfg), "concepts": [asdict(i) for i in infos]}
