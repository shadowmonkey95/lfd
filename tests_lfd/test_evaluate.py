"""BL-0006/0007/0008: post-hoc DiagonalAverage/PastConceptsAverage, cell-wise vs Naive, diagonal figure,
main table, upper-diagonal masking. Runs a tiny real sweep in a tmp project root (a few seconds)."""
import json
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from lfd import evaluate, naming, run
from lfd.metrics import DiagonalAverage, PastConceptsAverage


@pytest.fixture()
def tmp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "ROOT", tmp_path)
    monkeypatch.setattr(naming, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(naming, "DATA_PREP", tmp_path / "data" / "preprocess")
    evaluate._MATS.clear()
    rng = np.random.default_rng(0)
    tr, te = [], []
    for cid in range(3):
        c = np.full(3, 4.0 * cid)
        tr.append(pd.DataFrame(rng.normal(c, 1, (120, 3)), columns=list("abc")).assign(
            concept_id=cid, concept_name=f"C{cid}", label=0))
        x = np.vstack([rng.normal(c, 1, (60, 3)), rng.normal(c, 1, (10, 3)) + 5])
        te.append(pd.DataFrame(x, columns=list("abc")).assign(concept_id=cid, concept_name=f"C{cid}",
                                                              label=[0] * 60 + [1] * 10))
    d = naming.prep_dir("toy", "prep-v01")
    pd.concat(tr).to_parquet(naming.file_in(d, "train", "parquet"))
    pd.concat(te).to_parquet(naming.file_in(d, "test", "parquet"))
    cfg = {"name": "toy-grid", "dataset": "toy", "data_version": "prep-v01", "scenarios": ["concept_incremental"],
           "models": [{"name": "IsolationForest", "params": {"n_estimators": 20}}],
           "strategies": [{"name": "Naive"}, {"name": "Cumulative"},
                          {"name": "ReplayEnhanced", "params": {"buffer_size": 50}}],
           "seeds": [0, 1], "metrics": ["ROC-AUC", "AP"]}
    cp = tmp_path / "toy.yaml"
    cp.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(sys, "argv", ["lfd.run", "--config", str(cp)])
    run.main()
    return tmp_path


def _rows(root):
    idx = root / "results" / "toy" / "runs_index.jsonl"
    return idx, [json.loads(line) for line in idx.read_text().splitlines()]


def _evaluate(monkeypatch, *extra):
    evaluate._MATS.clear()
    monkeypatch.setattr(sys, "argv", ["lfd.evaluate", "--dataset", "toy", "--metric", "AP", *extra])
    evaluate.main()
    return sorted((naming.RESULTS / "toy" / "evaluations").iterdir())[-1]


def test_new_runs_store_new_metrics(tmp_root):
    _, rows = _rows(tmp_root)
    assert all(r["status"] == "ok" for r in rows) and len(rows) == 6
    assert all("AP__DiagonalAverage" in r and "ROC-AUC__PastConceptsAverage" in r for r in rows)


def test_posthoc_on_old_runs_and_outputs(tmp_root, monkeypatch):
    idx, rows = _rows(tmp_root)
    stored = {r["run_id"]: (r["AP__DiagonalAverage"], r["AP__PastConceptsAverage"]) for r in rows}
    # simulate runs made before 0.3.0: index has no new columns
    idx.write_text("".join(json.dumps({k: v for k, v in r.items() if "Diagonal" not in k and "PastConcepts" not in k})
                           + "\n" for r in rows))
    d = _evaluate(monkeypatch)
    f = lambda stem, ext="csv": naming.file_in(d, stem, ext)  # noqa: E731

    runs = pd.read_csv(f("runs"))
    for _, r in runs.iterrows():
        mat, _ = evaluate.run_matrix(r["path"], "concept_metric_callback_AP")
        assert r["AP__DiagonalAverage"] == pytest.approx(DiagonalAverage().compute(mat.tolist()))
        assert r["AP__PastConceptsAverage"] == pytest.approx(PastConceptsAverage().compute(mat.tolist()))
        assert r["AP__DiagonalAverage"] == pytest.approx(stored[r["run_id"]][0])  # post-hoc == stored at run time
        assert r["AP__PastConceptsAverage"] == pytest.approx(stored[r["run_id"]][1])

    comp = pd.read_csv(f("comparison"))
    assert {"AP__DiagonalAverage__mean", "AP__PastConceptsAverage__std"} <= set(comp.columns)
    assert len(json.loads(f("comparison", "json").read_text())) == len(comp)

    paired = pd.read_csv(f("paired_vs_naive"))
    assert {"AP__DiagonalAverage", "AP__PastConceptsAverage"} <= set(paired["metric"])

    cells = pd.read_csv(f("cellwise_vs_naive"))
    assert set(cells["config"]) == {"Cumulative", "ReplayEnhanced-buf50"}
    for _, r in cells.iterrows():
        assert r["better_mean"] + r["equal_mean"] + r["worse_mean"] == pytest.approx(r["n_cells"])
        assert r["n_cells"] == {"all": 6, "diagonal": 3, "past": 3}[r["region"]] and r["n_seeds"] == 2

    table = pd.read_csv(f("table_main"))
    assert {"DiagonalAverage__mean", "PastConceptsAverage__std", "BackwardTransfer__mean",
            "ForgettingMeasureStrict__mean", "cost_rel_cumulative", "n_seeds"} <= set(table.columns)
    assert table.loc[table["config"] == "Cumulative", "cost_rel_cumulative"].iloc[0] == pytest.approx(1.0)
    assert "DiagonalAverage" in f("table_main", "md").read_text()
    assert "PastConceptsAverage" in f("summary", "md").read_text()

    assert f("fig_diagonal_concept_incremental_IsolationForest", "png").exists()
    assert f("fig_heatmaps_concept_incremental_IsolationForest", "png").exists()


def test_cellwise_counts_hand_made(tmp_path, monkeypatch):
    monkeypatch.setattr(naming, "ROOT", tmp_path)
    evaluate._MATS.clear()
    mats = {"naive": [[0.5, 0.0], [0.5, 0.5]], "rep": [[0.6, 0.9], [0.502, 0.4]]}
    rows = []
    for name, m in mats.items():
        p = tmp_path / name
        p.mkdir()
        mm = {r: {c: m[i][j] for j, c in enumerate(["A", "B"])} for i, r in enumerate(["A", "B"])}
        (p / f"{name}__results.json").write_text(json.dumps(
            {"k": {"concepts_order": ["A", "B"], "metric_matrix": mm}}))
        rows.append({"config": "Naive" if name == "naive" else "Rep", "seed": 0, "path": name})
    out = {r["region"]: r for r in evaluate.cellwise_vs_naive(pd.DataFrame(rows), "k", 0.005)}
    # diagonal: +0.1 better, -0.1 worse; past: +0.002 equal; upper cell (0.9) ignored
    assert (out["all"]["better_mean"], out["all"]["equal_mean"], out["all"]["worse_mean"]) == (1, 1, 1)
    assert (out["diagonal"]["better_mean"], out["diagonal"]["worse_mean"]) == (1, 1)
    assert out["past"]["equal_mean"] == 1


def test_mask_upper():
    m = evaluate.mask_upper(np.ones((3, 3)))
    assert np.isnan(m[0, 1]) and np.isnan(m[0, 2]) and np.isnan(m[1, 2])
    assert not np.isnan(m[np.tril_indices(3)]).any()


def test_run_heatmap_hides_upper_by_default(tmp_root, monkeypatch):
    """lfd.run passes ignore_upper_diagonal=True unless the option is set (plots only)."""
    import pyclad.analysis.scenario_heatmap as hm

    seen = []
    orig = hm.plot_metric_heatmap
    monkeypatch.setattr(hm, "plot_metric_heatmap", lambda *a, **k: (seen.append(k["ignore_upper_diagonal"]),
                                                                     orig(*a, **k))[1])
    cfg = {"dataset": "toy", "scenarios": ["concept_incremental"], "metrics": ["AP"]}
    combo = {"scenario": "concept_incremental", "strategy": {"name": "Naive"},
             "model": {"name": "IsolationForest", "params": {"n_estimators": 5}}, "seed": 0}
    ds, ddir, tag = run.load_dataset("toy", "prep-v01")
    run.run_one(cfg, combo, "s", ds, ddir, tag)
    run.run_one({**cfg, "heatmap_show_upper_diagonal": True}, combo, "s", ds, ddir, tag)
    assert seen == [True, True, False, False]  # png + pdf per run
