"""Run an experiment grid with pyCLAD. Every (scenario x strategy x model x seed) = one run folder.

    python -m lfd.run --config configs/experiments/<dataset>/<name>.yaml [--dry-run]

Output per run (never overwritten):
  results/<dataset>/runs/<ts>__<dataset>__<scenario>__<strategy-params>__<model-params>__<prep-vNN>__s<seed>/
      <run_id>__results.json        raw pyCLAD output (metric matrices, time, memory, cost, [energy])
      <run_id>__config.json         fully-resolved config of THIS run (+ sweep id, data folder)
      <run_id>__env.json            python/package versions, git commit
      <run_id>__run.log             log
      <run_id>__heatmap_<metric>.png / .pdf
Append-only index: results/<dataset>/runs_index.jsonl  (one line per run, incl. failures)
"""
from __future__ import annotations

import argparse
import itertools
import json
import logging
import platform
import random
import subprocess
import sys
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from lfd import LFD_VERSION, features, naming, registry
from lfd.cost import CostCallback, CostLedger, CostTrackingModel
from lfd.proposals import require_approved


# ---------------------------------------------------------------- data
def load_dataset(raw_name: str, version: str):
    from pyclad.data.datasets.concepts_dataset import ConceptsDataset
    from pyclad.data.readers.concepts_readers import read_concepts_from_df

    d = naming.find_prep_dir(raw_name, version)
    train = pd.read_parquet(next(d.glob("*__train.parquet")))
    test = pd.read_parquet(next(d.glob("*__test.parquet")))
    feats = [c for c in train.columns if c not in ("concept_id", "concept_name", "label")]
    train[feats] = train[feats].astype("float32")
    test[feats] = test[feats].astype("float32")
    ds = ConceptsDataset(name=raw_name, train_concepts=read_concepts_from_df(train),
                         test_concepts=read_concepts_from_df(test))
    version_tag = d.name.split("__")[1]  # prep-vNN
    return ds, d, version_tag


# ---------------------------------------------------------------- env
def env_info() -> dict:
    pk = {}
    for p in ["pyclad", "pyod", "numpy", "scikit-learn", "pandas", "torch", "codecarbon"]:
        try:
            pk[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=naming.ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=naming.ROOT).strip())
    except Exception:
        commit, dirty = None, None
    return {"lfd_version": LFD_VERSION, "python": sys.version, "platform": platform.platform(), "packages": pk,
            "git_commit": commit, "git_dirty": dirty}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- one run
def run_one(cfg: dict, combo: dict, sweep: str, dataset, data_dir: Path, data_tag: str,
            feats: list | None = None, ftag: str = "") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pyclad.analysis.scenario_heatmap import plot_metric_heatmap
    from pyclad.callbacks.evaluation.concept_metric_evaluation import ConceptMetricCallback
    from pyclad.callbacks.evaluation.memory_usage import MemoryUsageCallback
    from pyclad.callbacks.evaluation.time_evaluation import TimeEvaluationCallback
    from pyclad.output.json_writer import JsonOutputWriter

    scen, strat, model, seed = combo["scenario"], combo["strategy"], combo["model"], combo["seed"]
    mparams = dict(model.get("params") or {})
    if model["name"] in registry.SEEDABLE:
        mparams.setdefault("random_state", seed)
    rid = naming.run_id(dataset=cfg["dataset"], scenario=scen, strategy=strat["name"],
                        strategy_params=strat.get("params"), model=model["name"],
                        model_params=model.get("params"), data_version=data_tag, seed=seed, features=ftag)
    d = naming.run_dir(cfg["dataset"], rid)
    rid = d.name  # may carry __r2 suffix

    log = logging.getLogger(rid)
    log.setLevel(logging.INFO)
    fh = logging.FileHandler(naming.file_in(d, "run", "log"))
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(fh)

    row = {"run_id": rid, "sweep_id": sweep, "dataset": cfg["dataset"], "data_folder": data_dir.name,
           "data_version": data_tag, "scenario": scen, "strategy": strat["name"],
           "strategy_params": strat.get("params") or {}, "model": model["name"], "model_params": mparams,
           "seed": seed, "features": [{"name": f.name, "params": p} for f, p in (feats or [])],
           "features_tag": ftag, "lfd_version": LFD_VERSION, "proposal": cfg.get("proposal"), "path": str(d.relative_to(naming.ROOT)),
           "status": "running"}
    naming.file_in(d, "config", "json").write_text(json.dumps({**row, "full_config": cfg}, indent=2, default=str))
    naming.file_in(d, "env", "json").write_text(json.dumps(env_info(), indent=2))

    try:
        seed_everything(seed)
        ledger = CostLedger()
        ctx = {"seed": seed, "scenario": scen, "strategy": strat["name"], "model": model["name"],
               "run_dir": d, "dataset": cfg["dataset"]}
        feats = feats or []
        run_dataset = dataset
        for f, p in feats:
            if f.transform_dataset:
                run_dataset = f.transform_dataset(run_dataset, p, ctx)
        # cost tracking sits closest to the real model, so it counts what is actually fitted
        factory = lambda: CostTrackingModel(registry.MODELS[model["name"]](**mparams), ledger)  # noqa: E731
        for f, p in feats:
            if f.wrap_model_factory:
                factory = f.wrap_model_factory(factory, p, ctx)
        strategy = registry.STRATEGIES[strat["name"]](factory, **(strat.get("params") or {}))
        for f, p in feats:
            if f.wrap_strategy:
                strategy = f.wrap_strategy(strategy, p, ctx)
        if not registry.compatible(strategy, scen):
            raise TypeError(f"{strat['name']} does not support {scen}")

        metric_cbs = {m: ConceptMetricCallback(base_metric=registry.BASE_METRICS[m](),
                                               summarized_metrics=[c() for c in registry.SUMMARY_METRICS])
                      for m in cfg.get("metrics", ["ROC-AUC", "AP"])}
        callbacks = [*metric_cbs.values(), TimeEvaluationCallback(), CostCallback(ledger)]
        if cfg.get("track_memory", False):  # tracemalloc: accurate but slows training a lot on big data
            callbacks.append(MemoryUsageCallback())
        if cfg.get("energy"):
            from pyclad.callbacks.evaluation.energy_evaluation import OfflineEnergyEvaluationCallback
            callbacks.append(OfflineEnergyEvaluationCallback(country_iso_code=cfg.get("energy_country", "VNM")))

        for f, p in feats:
            if f.callbacks:
                callbacks.extend(f.callbacks(p, ctx))

        scenario_cls = registry.SCENARIOS[scen][0]
        log.info("start %s", rid)
        scenario_cls(dataset=run_dataset, strategy=strategy, callbacks=callbacks).run()

        res_path = naming.file_in(d, "results", "json")
        first_model = factory()  # info provider for the model section (not fitted, not counted)
        JsonOutputWriter(res_path).write([first_model, run_dataset, strategy, *callbacks])
        res = json.loads(res_path.read_text())

        for m in metric_cbs:
            block = res[f"concept_metric_callback_{registry.BASE_METRICS[m]().name()}"]
            for k, v in block["metrics"].items():
                row[f"{m}__{k}"] = v
            order = block["concepts_order"]
            if set(block["metric_matrix"].keys()) == set(order):
                for ext in ("png", "pdf"):
                    plt.figure()
                    plot_metric_heatmap(block["metric_matrix"], order, annotate=len(order) <= 12,
                                        title=f"{m} | {scen} | {strat['name']} | {model['name']} | s{seed}",
                                        output_path=naming.file_in(d, f"heatmap_{naming.slug(m)}", ext))
                    plt.close("all")
        cost = res["cost_callback"]
        tm = res["time_evaluation_callback"]
        mem = res.get("memory_usage_callback", {}).get("peak_memory_usage", {})
        row.update({"train_time_s": tm["train_time_total"], "eval_time_s": tm["eval_time_total"],
                    "fit_seconds_total": cost["fit_seconds_total"],
                    "samples_fitted_total": cost["samples_fitted_total"],
                    "fit_calls_total": cost["fit_calls_total"], "models_fitted": cost["models_fitted"],
                    "peak_mem_mb": round(max(mem.values()) / 2**20, 3) if mem else None})
        if "energy_evaluation_callback" in res:
            row["energy_kwh"] = res["energy_evaluation_callback"]["emissions"].get("energy_consumed")
        row["status"] = "ok"
        log.info("done %s", rid)
    except Exception as e:  # keep going with the grid, but leave a visible trace
        row["status"] = "failed"
        row["error"] = f"{type(e).__name__}: {e}"
        naming.file_in(d, "ERROR", "txt").write_text(traceback.format_exc())
        log.error("FAILED %s: %s", rid, e)
    finally:
        root.removeHandler(fh)
        fh.close()

    idx = naming.RESULTS / naming.slug(cfg["dataset"], 40) / "runs_index.jsonl"
    with open(idx, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
    return row


# ---------------------------------------------------------------- grid
def expand(cfg: dict) -> list[dict]:
    combos, skipped = [], set()
    for scen, strat, model, seed in itertools.product(cfg["scenarios"], cfg["strategies"], cfg["models"],
                                                      cfg.get("seeds", [0])):
        if not registry.compatible_name(strat["name"], scen):
            if (strat["name"], scen) not in skipped:
                print(f"skip: {strat['name']} does not support {scen} (by design in pyCLAD)")
                skipped.add((strat["name"], scen))
            continue
        combos.append({"scenario": scen, "strategy": strat, "model": model, "seed": seed})
    return combos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="list runs and planned names, run nothing")
    ap.add_argument("--feature", action="append", metavar="NAME[:k=v,...]",
                    help="enable an opt-in feature for this sweep (repeatable; overrides the config's entry). "
                         "List them with: python -m lfd.features")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = yaml.safe_load(Path(args.config).read_text())
    if cfg.get("proposal"):
        require_approved(naming.ROOT / cfg["proposal"], kind="improvement")
    for k in ("dataset", "scenarios", "strategies", "models"):
        if k not in cfg:
            sys.exit(f"config missing '{k}'")
    for s in cfg["strategies"]:
        if s["name"] not in registry.STRATEGIES:
            sys.exit(f"unknown strategy {s['name']}; known: {list(registry.STRATEGIES)}")
    for m in cfg["models"]:
        if m["name"] not in registry.MODELS:
            sys.exit(f"unknown model {m['name']}; known: {list(registry.MODELS)}")

    specs = features.merge(cfg.get("features"), features.parse_cli(args.feature))
    feats = features.resolve(specs)
    ftag = features.tag(feats)
    cfg["features"] = specs  # effective features (config + CLI), saved with every run
    if feats:
        print("features:", ", ".join(f"{f.name}{p}" for f, p in feats))

    combos = expand(cfg)
    dataset, data_dir, data_tag = load_dataset(cfg["dataset"], cfg.get("data_version", "latest"))
    sweep = naming.sweep_id(cfg.get("name", Path(args.config).stem))
    print(f"sweep {sweep}: {len(combos)} runs on {data_dir.name}")
    if args.dry_run:
        for c in combos:
            print("  ", naming.run_id(dataset=cfg["dataset"], scenario=c["scenario"], strategy=c["strategy"]["name"],
                                      strategy_params=c["strategy"].get("params"), model=c["model"]["name"],
                                      model_params=c["model"].get("params"), data_version=data_tag, seed=c["seed"],
                                      features=ftag))
        return

    # keep a copy of the exact config that launched the sweep
    sweeps_dir = naming.RESULTS / naming.slug(cfg["dataset"], 40) / "sweeps"
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    (sweeps_dir / f"{sweep}.yaml").write_text(
        f"# source: {args.config}\n# effective config incl. --feature overrides, lfd {LFD_VERSION}\n"
        + yaml.safe_dump(cfg, sort_keys=False))

    rows = [run_one(cfg, c, sweep, dataset, data_dir, data_tag, feats, ftag) for c in combos]
    ok = sum(r["status"] == "ok" for r in rows)
    print(f"\n{ok}/{len(rows)} runs ok. sweep id: {sweep}")
    for r in rows:
        if r["status"] != "ok":
            print(f"  FAILED {r['run_id']}: {r.get('error')}")


if __name__ == "__main__":
    main()
