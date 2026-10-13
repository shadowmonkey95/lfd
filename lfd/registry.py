"""Name -> pyCLAD object builders. Configs refer to these names; add new ones here.

To add a torch-based strategy (EWC, LwF, A-GEM, DER++, VLAD) register a builder that
constructs the backbone + runner as in pyCLAD's examples/strategies/*.py, and wrap the model with
CostTrackingModel only if it exposes fit(); otherwise record cost from the runner.
"""
from __future__ import annotations

from typing import Callable, Dict

from pyclad.models.adapters.pyod_adapters import (
    COPODAdapter, ECODAdapter, IsolationForestAdapter, LocalOutlierFactorAdapter, OneClassSVMAdapter,
)
from pyclad.models.model import Model
from pyclad.scenarios.concept_agnostic import ConceptAgnosticScenario
from pyclad.scenarios.concept_aware import ConceptAwareScenario
from pyclad.scenarios.concept_incremental import ConceptIncrementalScenario
from pyclad.strategies.baselines.cumulative import CumulativeStrategy
from pyclad.strategies.baselines.mste import MSTE
from pyclad.strategies.baselines.naive import NaiveStrategy
from pyclad.strategies.replay.buffers.adaptive_balanced import AdaptiveBalancedReplayBuffer
from pyclad.strategies.replay.replay import ReplayEnhancedStrategy, ReplayOnlyStrategy
from pyclad.strategies.replay.selection.random import RandomSelection
from pyclad.strategies.strategy import ConceptAgnosticStrategy, ConceptAwareStrategy, ConceptIncrementalStrategy

from pyclad.metrics.base.average_precision import AveragePrecision
from pyclad.metrics.base.f1_score import F1Score
from pyclad.metrics.base.roc_auc import RocAuc
from pyclad.metrics.continual.average_continual import ContinualAverage
from pyclad.metrics.continual.backward_transfer import BackwardTransfer
from pyclad.metrics.continual.final_step_average import FinalStepAverage
from pyclad.metrics.continual.forgetting_measure_strict import ForgettingMeasureStrict
from pyclad.metrics.continual.forward_transfer import ForwardTransfer

from lfd.metrics import DiagonalAverage, PastConceptsAverage

# ---------------- models ----------------
SEEDABLE = {"IsolationForest", "AE", "VAE"}  # models that accept random_state


def _pyod_deep(kind: str, **p) -> Model:
    """pyOD deep models (need torch). Imported lazily so torch stays optional."""
    from pyclad.models.adapters.pyod_adapters import PyODAdapter

    p.setdefault("verbose", 0)
    if kind == "AE":
        from pyod.models.auto_encoder import AutoEncoder
        return PyODAdapter(AutoEncoder(**p), model_name="AE")
    from pyod.models.vae import VAE
    return PyODAdapter(VAE(**p), model_name="VAE")


MODELS: Dict[str, Callable[..., Model]] = {
    "IsolationForest": lambda **p: IsolationForestAdapter(**p),
    "LOF": lambda **p: LocalOutlierFactorAdapter(**p),
    "OCSVM": lambda **p: OneClassSVMAdapter(**p),
    "COPOD": lambda **p: COPODAdapter(**p),
    "ECOD": lambda **p: ECODAdapter(**p),
    # legacy lfd: AE(hidden_neuron_list=[32, 24, 16], epoch_num=20); VAE([32,24,16]/[16,24,32], latent_dim=8,
    # epoch_num=20, preprocessing=False). Pass these as params in the sweep config.
    "AE": lambda **p: _pyod_deep("AE", **p),
    "VAE": lambda **p: _pyod_deep("VAE", **p),
}

# ---------------- strategies ----------------
# each builder: (model_factory, **params) -> strategy. model_factory() returns a fresh (cost-tracked) model.


def _buffer(p):
    return AdaptiveBalancedReplayBuffer(RandomSelection(), max_size=int(p.get("buffer_size", 500)))


STRATEGIES: Dict[str, Callable] = {
    "Naive": lambda mf, **p: NaiveStrategy(mf()),
    "Cumulative": lambda mf, **p: CumulativeStrategy(mf()),
    "MSTE": lambda mf, **p: MSTE(mf),
    "ReplayOnly": lambda mf, **p: ReplayOnlyStrategy(mf(), _buffer(p)),
    "ReplayEnhanced": lambda mf, **p: ReplayEnhancedStrategy(mf(), _buffer(p)),
}

# ---------------- scenarios ----------------
SCENARIOS = {
    "concept_incremental": (ConceptIncrementalScenario, ConceptIncrementalStrategy),
    "concept_aware": (ConceptAwareScenario, ConceptAwareStrategy),
    "concept_agnostic": (ConceptAgnosticScenario, ConceptAgnosticStrategy),
}

# ---------------- metrics ----------------
BASE_METRICS = {"ROC-AUC": RocAuc, "AP": AveragePrecision, "F1": F1Score}
SUMMARY_METRICS = [ContinualAverage, BackwardTransfer, ForwardTransfer, ForgettingMeasureStrict, FinalStepAverage,
                   DiagonalAverage, PastConceptsAverage]  # last two: BL-0005 (lfd/metrics.py)


STRATEGY_CLASSES = {"Naive": NaiveStrategy, "Cumulative": CumulativeStrategy, "MSTE": MSTE,
                    "ReplayOnly": ReplayOnlyStrategy, "ReplayEnhanced": ReplayEnhancedStrategy}


def compatible(strategy, scenario: str) -> bool:
    return isinstance(strategy, SCENARIOS[scenario][1])


def compatible_name(strategy_name: str, scenario: str) -> bool:
    cls = STRATEGY_CLASSES.get(strategy_name)
    return cls is None or issubclass(cls, SCENARIOS[scenario][1])  # unknown class -> checked at runtime
