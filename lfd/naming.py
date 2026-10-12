"""Naming + path conventions. Every output path in this project MUST come from here.

Guarantees:
  * every name carries dataset, scenario, strategy (+params), model, data version, seed and a timestamp
  * directories are created with exist_ok=False -> a rerun can never overwrite an earlier result
    (on a same-second collision a suffix __r2, __r3 ... is appended)
"""
from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PREP = ROOT / "data" / "preprocess"
RESULTS = ROOT / "results"
RESEARCH = ROOT / "research"  # proposals, backlog, meetings, logs, changelog
PROPOSALS = RESEARCH / "proposals"
PIPELINES = ROOT / "pipelines"

SCENARIO_CODES = {
    "concept_incremental": "incremental",
    "concept_aware": "aware",
    "concept_agnostic": "agnostic",
}


def timestamp() -> str:
    """Local time, sortable, filesystem-safe: 20261012-000612"""
    return _dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def slug(text: str, max_len: int = 60) -> str:
    text = re.sub(r"[^A-Za-z0-9.+-]+", "-", str(text)).strip("-")
    return text[:max_len] or "x"


def strategy_tag(name: str, params: dict | None) -> str:
    """ReplayEnhanced + {buffer_size: 500, selection: random} -> ReplayEnhanced-buf500-random"""
    abbrev = {"buffer_size": "buf", "max_size": "buf", "lambda_ewc": "lam", "alpha": "a", "selection": ""}
    parts = [name]
    for k, v in sorted((params or {}).items()):
        parts.append(f"{abbrev.get(k, slug(k, 6))}{slug(v, 12)}")
    return slug("-".join(parts), 60)


def model_tag(name: str, params: dict | None) -> str:
    abbrev = {"n_estimators": "n", "contamination": "cont", "n_neighbors": "k", "max_samples": "ms",
              "kernel": "", "nu": "nu", "random_state": None}
    parts = [name] + [f"{abbrev.get(k, slug(k, 6))}{slug(v, 10)}" for k, v in sorted((params or {}).items())
                      if abbrev.get(k, "") is not None]
    return slug("-".join(parts), 50)


def run_id(*, dataset: str, scenario: str, strategy: str, strategy_params: dict | None,
           model: str, model_params: dict | None, data_version: str, seed: int, ts: str | None = None,
           features: str = "") -> str:
    """features: tag from lfd.features.tag(); empty when no opt-in feature is enabled."""
    return "__".join([
        ts or timestamp(),
        slug(dataset, 40),
        SCENARIO_CODES.get(scenario, slug(scenario)),
        strategy_tag(strategy, strategy_params),
        model_tag(model, model_params),
        *([features] if features else []),
        slug(data_version, 20),
        f"s{seed}",
    ])


def make_unique_dir(path: Path) -> Path:
    """Create `path`; if it exists, create path__r2, path__r3 ... Never reuses a directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate, i = path, 1
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            i += 1
            candidate = path.with_name(f"{path.name}__r{i}")


def run_dir(dataset: str, rid: str) -> Path:
    return make_unique_dir(RESULTS / slug(dataset, 40) / "runs" / rid)


def sweep_id(config_name: str, ts: str | None = None) -> str:
    return f"{ts or timestamp()}__sweep__{slug(config_name, 50)}"


def eval_dir(dataset: str, scope: str) -> Path:
    return make_unique_dir(RESULTS / slug(dataset, 40) / "evaluations" / f"{timestamp()}__eval__{scope}")


def next_prep_version(raw_name: str) -> str:
    """prep-v01, prep-v02 ... based on what already exists in data/preprocess/<raw_name>/"""
    base = DATA_PREP / raw_name
    nums = [int(m.group(1)) for p in base.glob(f"{raw_name}__prep-v*")
            if (m := re.search(r"__prep-v(\d+)__", p.name))]
    return f"prep-v{(max(nums) + 1) if nums else 1:02d}"


def prep_dir(raw_name: str, version: str) -> Path:
    return make_unique_dir(DATA_PREP / raw_name / f"{raw_name}__{version}__{timestamp()}")


def find_prep_dir(raw_name: str, version: str = "latest") -> Path:
    """Resolve a preprocessed dataset folder. version='latest' or e.g. 'prep-v02'."""
    base = DATA_PREP / raw_name
    candidates = sorted(p for p in base.glob(f"{raw_name}__prep-v*") if p.is_dir())
    if version != "latest":
        candidates = [p for p in candidates if f"__{version}__" in p.name]
    if not candidates:
        raise FileNotFoundError(f"No preprocessed data for '{raw_name}' version '{version}' in {base}")
    return candidates[-1]


def file_in(directory: Path, stem: str, suffix: str) -> Path:
    """Files inside a run/eval folder repeat the folder id so they stay unique if copied elsewhere:
    <folder-id>__<stem>.<suffix>"""
    return directory / f"{directory.name}__{stem}.{suffix.lstrip('.')}"
