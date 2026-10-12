"""Fast tests for the research harness. Run: pytest tests_lfd -q
Every new opt-in feature or lfd change must add a test here (or in a new tests_lfd/test_<name>.py)."""
import numpy as np
import pytest

from lfd import features, naming
from pyclad.data.concept import Concept
from pyclad.data.datasets.concepts_dataset import ConceptsDataset


def _ds(n=3):
    tr = [Concept(f"C{i}", np.zeros((5, 2))) for i in range(n)]
    te = [Concept(f"C{i}", np.zeros((5, 2)), np.array([0, 0, 0, 1, 1])) for i in range(n)]
    return ConceptsDataset("t", tr, te)


def test_make_unique_dir_never_reuses(tmp_path):
    a = naming.make_unique_dir(tmp_path / "x")
    b = naming.make_unique_dir(tmp_path / "x")
    assert a != b and b.name == "x__r2"


def test_run_id_contains_everything():
    rid = naming.run_id(dataset="ieee", scenario="concept_incremental", strategy="ReplayEnhanced",
                        strategy_params={"buffer_size": 200}, model="IsolationForest",
                        model_params={"n_estimators": 100}, data_version="prep-v02", seed=3, ts="20260101-000000",
                        features="ft-conceptorder")
    assert rid == ("20260101-000000__ieee__incremental__ReplayEnhanced-buf200__IsolationForest-n100"
                   "__ft-conceptorder__prep-v02__s3")


def test_feature_cli_parse_and_merge():
    cli = features.parse_cli(["concept_order:order=shuffle,seed=4"])
    merged = features.merge([{"name": "concept_order", "params": {"order": "reverse"}}], cli)
    assert merged == [{"name": "concept_order", "params": {"order": "shuffle", "seed": 4}}]


def test_feature_unknown_param_rejected():
    with pytest.raises(SystemExit):
        features.resolve([{"name": "concept_order", "params": {"nope": 1}}])


def test_feature_tag_empty_when_disabled():
    assert features.tag([]) == ""


def test_concept_order_reverse_and_custom():
    f = features.FEATURES["concept_order"]
    out = f.transform_dataset(_ds(), {**f.defaults, "order": "reverse"}, {})
    assert [c.name for c in out.train_concepts()] == ["C2", "C1", "C0"]
    assert [c.name for c in out.test_concepts()] == ["C2", "C1", "C0"]
    out = f.transform_dataset(_ds(), {**f.defaults, "order": "custom", "names": "C1|C2|C0"}, {})
    assert [c.name for c in out.train_concepts()] == ["C1", "C2", "C0"]


def test_every_feature_has_backlog_id_and_version():
    for f in features.FEATURES.values():
        assert f.backlog_id and f.since, f"feature {f.name} must declare backlog_id and since"


def test_concept_order_shuffle_never_identity():
    f = features.FEATURES["concept_order"]
    for seed in range(20):
        out = f.transform_dataset(_ds(4), {**f.defaults, "order": "shuffle", "seed": seed}, {})
        assert [c.name for c in out.train_concepts()] != ["C0", "C1", "C2", "C3"]
