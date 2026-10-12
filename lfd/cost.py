"""Cost accounting — the "saving costs" half of the research claim.

pyCLAD's TimeEvaluationCallback measures wall-clock; that is noisy and machine-dependent.
Here we add hardware-independent cost: how many samples each strategy actually re-fits per step.
  * Naive          fits only the new concept         -> cheapest, forgets most
  * Cumulative     re-fits ALL data seen so far       -> upper bound on cost (and usually on retention)
  * Replay*        fits new data + a bounded buffer   -> the trade-off you want to show
  * MSTE           one model per concept              -> cheap fits, but memory grows (models_alive)

Wrap the model with CostTrackingModel (MSTE: wrap the factory) and add CostCallback to callbacks.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List

import numpy as np

from pyclad.callbacks.callback import Callback
from pyclad.models.model import Model
from pyclad.output.output_writer import InfoProvider


class CostLedger:
    def __init__(self):
        self.step: str | None = None
        self.events: List[dict] = []


class CostTrackingModel(Model):
    def __init__(self, inner: Model, ledger: CostLedger):
        self._inner = inner
        self._ledger = ledger

    def fit(self, data: np.ndarray):
        t0 = time.perf_counter()
        out = self._inner.fit(data)
        self._ledger.events.append({"step": self._ledger.step, "model": id(self), "samples": int(len(data)),
                                    "seconds": time.perf_counter() - t0})
        return out

    def predict(self, data: np.ndarray):
        return self._inner.predict(data)

    def name(self) -> str:
        return self._inner.name()

    def info(self) -> Dict[str, Any]:
        return self._inner.info()

    def additional_info(self):
        return self._inner.additional_info()


def tracking_factory(factory: Callable[[], Model], ledger: CostLedger) -> Callable[[], Model]:
    return lambda: CostTrackingModel(factory(), ledger)


class CostCallback(Callback, InfoProvider):
    def __init__(self, ledger: CostLedger):
        self._ledger = ledger
        self._order: List[str] = []

    def before_concept_processing(self, concept=None, **kwargs):
        self._ledger.step = concept.name
        self._order.append(concept.name)

    def info(self) -> Dict[str, Any]:
        per_step = []
        cumulative = 0
        for name in self._order:
            ev = [e for e in self._ledger.events if e["step"] == name]
            s = sum(e["samples"] for e in ev)
            cumulative += s
            per_step.append({"concept": name, "fit_calls": len(ev), "samples_fitted": s,
                             "samples_fitted_cumulative": cumulative,
                             "fit_seconds": round(sum(e["seconds"] for e in ev), 4)})
        return {"cost_callback": {
            "per_step": per_step,
            "samples_fitted_total": cumulative,
            "fit_calls_total": len(self._ledger.events),
            "fit_seconds_total": round(sum(e["seconds"] for e in self._ledger.events), 4),
            "models_fitted": len({e["model"] for e in self._ledger.events}),  # >1 only for MSTE-like
        }}
