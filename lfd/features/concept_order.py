"""Opt-in: change the order in which concepts are learned (and evaluated), to test robustness of
forgetting/cost conclusions to the stream order. Default pipeline order is untouched unless enabled."""
from __future__ import annotations

import numpy as np

from pyclad.data.datasets.concepts_dataset import ConceptsDataset

from lfd.features import Feature, register


def _transform(dataset: ConceptsDataset, params: dict, ctx: dict) -> ConceptsDataset:
    train = list(dataset.train_concepts())
    n = len(train)
    order = params["order"]
    if order == "reverse":
        idx = list(range(n))[::-1]
    elif order == "shuffle":
        rng = np.random.default_rng(params["seed"])
        idx = list(rng.permutation(n))
        while n > 1 and idx == list(range(n)):  # a "shuffle" must actually change the order
            idx = list(rng.permutation(n))
    elif order == "custom":
        names = [c.name for c in train]
        idx = [names.index(x) for x in str(params["names"]).split("|")]
        if sorted(idx) != list(range(n)):
            raise ValueError(f"custom order must list every concept exactly once: {names}")
    else:
        raise ValueError(f"order must be reverse|shuffle|custom, got {order}")
    new_train = [train[i] for i in idx]
    by_name = {c.name: c for c in dataset.test_concepts()}
    new_test = [by_name[c.name] for c in new_train if c.name in by_name] + \
               [c for c in dataset.test_concepts() if c.name not in {t.name for t in new_train}]
    return ConceptsDataset(name=dataset.name(), train_concepts=new_train, test_concepts=new_test)


register(Feature(
    name="concept_order",
    description="Reorder the concept stream: reverse | shuffle (seeded) | custom (names='C2|C0|C1').",
    defaults={"order": "reverse", "seed": 0, "names": ""},
    backlog_id="BL-0000",
    since="0.1.0",
    transform_dataset=_transform,
))
