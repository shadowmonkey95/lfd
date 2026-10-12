"""IEEE-CIS Fraud Detection -> time-based continual stream. prep_v01 = faithful port of the legacy lfd setup
(tag legacy-v1: src/main.py with scenario_config="time_based", concepts_no=8, size_per_concept=73817,
data from ead_preprocess_keepTransactionDT.py). Use it to reproduce old results, then improve in prep_v02+.

Raw files expected in data/raw/ieee-fraud-detection/: train_transaction.csv, train_identity.csv
(Kaggle competition download, unchanged).
"""
from pathlib import Path

from lfd.datasets import ieee_ead
from lfd.streams.time_windows import TimeStreamConfig, build_time_concepts, describe

# Every knob of this version. Changing any value = new prep version + new proposal.
PARAMS = {
    "use_identity": True,
    "low_info_threshold": 0.9,
    "stream": TimeStreamConfig(
        n_concepts=8,
        size_per_concept=73817,     # legacy value; effective size = min(this, normals // 8) = 71234
        window="equal_count",
        tail="drop",
        train_ratio=0.6,
        anomaly_span="window",
        anomaly_pick="earliest",
        anomaly_cap=-1,             # legacy rule: min(n_fraud // n_concepts, 0.2 * size_per_concept)
        nearest_fallback=True,
    ),
}


def process(raw_dir: Path) -> dict:
    df, report = ieee_ead.preprocess(raw_dir, PARAMS["use_identity"], PARAMS["low_info_threshold"])
    before = len(df)
    df = df.dropna()  # legacy load_csv_dataset did this
    report["rows_dropped_na"] = before - len(df)
    cfg = PARAMS["stream"]
    train, test, infos = build_time_concepts(df, time_col=ieee_ead.TIME_COL, label_col=ieee_ead.LABEL_COL, cfg=cfg)
    cols = ["concept_id", "concept_name", "label"] + [c for c in train.columns if c not in ("concept_id", "concept_name", "label")]
    return {
        "train": train[cols],
        "test": test[cols],
        "params": {"use_identity": PARAMS["use_identity"], "low_info_threshold": PARAMS["low_info_threshold"],
                   **describe(cfg, infos), "ead_report": report},
        "notes": ("Faithful port of legacy-v1 time_based scenario. Known caveats (candidates for prep_v02): "
                  "(1) scaler/encoders/medians/group stats fit on ALL rows incl. future concepts; "
                  "(2) test anomalies are the EARLIEST frauds of each window, i.e. from the training period, while "
                  "test normals come from the end of the window; (3) time column removed from features."),
    }
