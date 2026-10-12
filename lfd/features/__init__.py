"""Opt-in features: improvements that change training/evaluation behaviour stay OFF unless a sweep asks for them.

Enable in a sweep config:
    features:
      - {name: concept_order, params: {order: reverse}}
or on the command line (overrides the config for that feature name):
    python -m lfd.run --config ... --feature concept_order:order=reverse,seed=3

Every enabled feature (+ its non-default params) becomes part of the run id (`__ft-...`) and of the
evaluation grouping, so runs with and without a feature are never mixed or overwritten.

Writing a feature: add lfd/features/<name>.py that calls `register(Feature(...))`. Hooks (all optional):
    transform_dataset(dataset, params, ctx) -> dataset        before the scenario starts
    wrap_model_factory(factory, params, ctx) -> factory       models are built through this
    wrap_strategy(strategy, params, ctx) -> strategy          after the strategy is built
    callbacks(params, ctx) -> list[Callback]                  extra callbacks (results land in __results.json)
ctx = {"seed", "scenario", "strategy", "model", "run_dir", "dataset"}
Each feature module must have a test in tests_lfd/ and a CHANGELOG entry.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from lfd import naming


@dataclass
class Feature:
    name: str
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)
    backlog_id: str = ""
    since: str = ""  # lfd version that introduced it
    transform_dataset: Optional[Callable] = None
    wrap_model_factory: Optional[Callable] = None
    wrap_strategy: Optional[Callable] = None
    callbacks: Optional[Callable] = None


FEATURES: Dict[str, Feature] = {}


def register(feature: Feature) -> Feature:
    if feature.name in FEATURES:
        raise ValueError(f"feature '{feature.name}' registered twice")
    if not feature.name.replace("_", "").isalnum():
        raise ValueError(f"feature name must be snake_case: {feature.name}")
    FEATURES[feature.name] = feature
    return feature


def _autoload() -> None:
    for mod in pkgutil.iter_modules(__path__):
        if not mod.name.startswith("_"):
            importlib.import_module(f"{__name__}.{mod.name}")


def _coerce(v: str):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False, "none": None}.get(v.lower(), v)


def parse_cli(items: List[str] | None) -> List[dict]:
    """['concept_order:order=reverse,seed=3', 'x'] -> [{'name': 'concept_order', 'params': {...}}, {'name': 'x'}]"""
    out = []
    for item in items or []:
        name, _, rest = item.partition(":")
        params = {}
        for kv in filter(None, rest.split(",")):
            k, _, v = kv.partition("=")
            params[k.strip()] = _coerce(v.strip())
        out.append({"name": name.strip(), "params": params})
    return out


def merge(config_specs: List[dict] | None, cli_specs: List[dict]) -> List[dict]:
    merged = {s["name"]: {"name": s["name"], "params": dict(s.get("params") or {})} for s in (config_specs or [])}
    for s in cli_specs:
        merged[s["name"]] = s
    return sorted(merged.values(), key=lambda s: s["name"])


def resolve(specs: List[dict]) -> List[tuple[Feature, dict]]:
    """Validate names/params and fill defaults. Raises on unknown feature or param."""
    resolved = []
    for s in specs:
        f = FEATURES.get(s["name"])
        if f is None:
            raise SystemExit(f"unknown feature '{s['name']}'. Available: {sorted(FEATURES)}")
        unknown = set(s.get("params") or {}) - set(f.defaults)
        if unknown:
            raise SystemExit(f"feature '{f.name}' has no params {sorted(unknown)}; allowed: {f.defaults}")
        resolved.append((f, {**f.defaults, **(s.get("params") or {})}))
    return resolved


def tag(resolved: List[tuple[Feature, dict]]) -> str:
    """'' when nothing is enabled, else 'ft-concept_order-reverse' (only non-default params are spelled out)."""
    if not resolved:
        return ""
    parts = []
    for f, p in resolved:
        diffs = [naming.slug(v, 10) for k, v in sorted(p.items()) if f.defaults.get(k) != v]
        parts.append("-".join([f.name.replace("_", "")] + diffs))
    return naming.slug("ft-" + "+".join(parts), 50)


def describe() -> str:
    lines = []
    for name, f in sorted(FEATURES.items()):
        lines.append(f"{name}  [{f.backlog_id or '-'} since {f.since or '?'}]\n    {f.description}\n    params: {f.defaults}")
    return "\n".join(lines) or "(no features registered)"


_autoload()

if __name__ == "__main__":
    print(describe())
