"""Tests for lfd.streams.time_windows (time-based concept streams)."""
import numpy as np
import pandas as pd
import pytest

from lfd.streams.time_windows import TimeStreamConfig, build_time_concepts


def _df(n=1000, fraud_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(0, 10_000, n)).astype(float)
    y = (rng.random(n) < fraud_rate).astype(int)
    return pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n), "TransactionDT": t, "isFraud": y})


def _build(df, **kw):
    return build_time_concepts(df, "TransactionDT", "isFraud", TimeStreamConfig(**kw))


def test_schema_and_time_column_removed():
    tr, te, _ = _build(_df(), n_concepts=4)
    assert list(tr.columns) == ["concept_id", "concept_name", "label", "f1", "f2"]
    assert list(te.columns) == list(tr.columns)
    assert (tr["label"] == 0).all()
    assert set(te["label"].unique()) == {0, 1}


def test_legacy_equal_count_drop_sizes_and_split():
    df = _df()
    n_norm = int((df.isFraud == 0).sum())
    tr, te, infos = _build(df, n_concepts=4)
    step = n_norm // 4
    for info in infos:
        assert info.train_normals == int(0.6 * step)
        assert info.test_normals == step - int(0.6 * step)
    assert len(infos) == 4


def test_concepts_are_time_ordered_and_disjoint():
    _, _, infos = _build(_df(), n_concepts=5)
    for a, b in zip(infos, infos[1:]):
        assert a.time_end <= b.time_start


@pytest.mark.parametrize("tail,expected", [("drop", 3), ("include", 3), ("overflow", 4)])
def test_tail_strategies(tail, expected):
    df = _df(n=1000, fraud_rate=0.0)
    df.loc[:9, "isFraud"] = 1
    # 990 normals, size_per_concept 300 -> 3 windows of 300, 90 left (< half) for overflow... make it > half
    _, _, infos = _build(df, n_concepts=3, size_per_concept=200, tail=tail)
    assert len(infos) == expected
    if tail == "include":
        assert infos[-1].train_normals + infos[-1].test_normals == 990 - 2 * 200


def test_test_period_anomalies_inside_test_span():
    df = _df(n=4000, fraud_rate=0.1)
    _, te, infos = _build(df, n_concepts=4, anomaly_span="test_period", anomaly_cap=None)
    assert all(i.test_anomalies == i.anomalies_in_span for i in infos)


def test_equal_time_windows_cover_equal_duration():
    _, _, infos = _build(_df(n=3000), n_concepts=3, window="equal_time")
    spans = [i.time_end - i.time_start for i in infos]
    assert max(spans) - min(spans) < 0.05 * 10_000


def test_explicit_cap_and_spread_pick():
    tr, te, infos = _build(_df(n=4000, fraud_rate=0.2), n_concepts=2, anomaly_cap=10, anomaly_pick="spread")
    assert all(i.test_anomalies == 10 for i in infos)


def test_registry_has_deep_models():
    from lfd import registry
    assert {"AE", "VAE"} <= set(registry.MODELS)
    torch = pytest.importorskip("torch")  # noqa: F841
    m = registry.MODELS["AE"](hidden_neuron_list=[4], epoch_num=1, random_state=0)
    m.fit(np.random.rand(64, 3).astype("float32"))
    assert m.predict(np.random.rand(8, 3).astype("float32")).anomaly_scores.shape == (8,)
