# Running experiments: a guide for new students

*Backlog: BL-0010. For the dataset/stream itself see `SCENARIOS_AND_DATASETS.md`. For the Claude Code workflow see
`RESEARCH.md` §3.*

The project asks one question: **do continual strategies keep performance on old fraud patterns (close to
`Cumulative`, which re-trains on everything) while costing about as little as `Naive` (which trains on new data
only)?** Every sweep includes `Naive` and `Cumulative` as the two reference points.

## 1. Setup (with uv)

```bash
git clone https://github.com/shadowmonkey95/lfd.git && cd lfd
git remote add upstream https://github.com/lifelonglab/pyCLAD.git
uv venv .venv --python 3.12
source .venv/bin/activate                     # Windows: .venv\Scripts\activate
uv pip install -e .                           # pyCLAD itself (editable, from src/)
uv pip install -r requirements-lfd.txt        # harness dependencies
uv pip install torch                          # only for AE / VAE models
pytest tests_lfd -q                           # harness tests; pyCLAD's own: pytest tests -q
```

Use `uv pip install` only. **Do not** run `uv add`, `uv sync` or `uv init`: they rewrite pyCLAD's `pyproject.toml`
or create a lockfile, and that breaks clean upgrades of pyCLAD. If you see "No module named …", the venv is not
active; use `.venv/bin/python -m …`.

Put the Kaggle files, unchanged, in `data/raw/ieee-fraud-detection/` (`train_transaction.csv`, `train_identity.csv`).

## 2. The workflow in four commands

```bash
# 1) preprocess (only with an APPROVED data proposal: the approval gate)
python -m lfd.prep --raw ieee-fraud-detection --pipeline pipelines/ieee-fraud-detection/prep_v01.py \
                   --proposal research/proposals/ieee-fraud-detection/<approved>.md
# 2) train a sweep (always dry-run first: it lists every run and its folder name)
python -m lfd.run --config configs/experiments/ieee-fraud-detection/legacy-timebased-grid.yaml --dry-run
python -m lfd.run --config configs/experiments/ieee-fraud-detection/legacy-timebased-grid.yaml
# 3) evaluate (once per base metric; AP first for fraud)
python -m lfd.evaluate --dataset ieee-fraud-detection --sweep <sweep id printed by lfd.run> --metric AP
python -m lfd.evaluate --dataset ieee-fraud-detection --sweep <sweep id> --metric ROC-AUC
# 4) where am I?
python -m lfd.status
```

Useful extras:
- `python -m lfd.features`: lists the opt-in switches. Enable one per sweep with `features:` in the YAML or
  `--feature name:k=v`.
- `python -m lfd.proposals list --status pending`: proposals waiting for approval.
- Heatmaps hide the cells above the diagonal (concepts not learned yet). To show them, use
  `heatmap_show_upper_diagonal: true` in the sweep YAML or `--heatmap-show-upper-diagonal` for `lfd.run`, and
  `--show-upper-diagonal` for `lfd.evaluate`. This changes plots only.

## 3. A sweep config

`configs/experiments/<dataset>/<name>.yaml`: one file is one sweep. Every scenario × strategy × model × seed becomes
one run. Incompatible pairs are skipped (e.g. `MSTE` needs `concept_aware`).

```yaml
name: my-sweep
dataset: ieee-fraud-detection
data_version: prep-v01            # pin the preprocessing version
proposal: null                    # path to an approved improvement proposal for non-baseline sweeps
scenarios: [concept_incremental]
models: [{name: IsolationForest}]
strategies: [{name: Naive}, {name: Cumulative}, {name: ReplayEnhanced, params: {buffer_size: 1000}}]
seeds: [0, 1, 2, 3, 4]            # >=3; 5 for anything in the paper
metrics: [ROC-AUC, AP]
features: []
```

Never edit a config that has already run; copy it to a new name. The exact config of every launched sweep is saved
in `results/<dataset>/sweeps/`.

## 4. Where things are (and the never-overwrite rule)

All paths come from `lfd/naming.py`. Every folder name carries a timestamp, so nothing is ever overwritten, and
every file inside a folder starts with the folder name.

```
data/raw/<dataset>/                               original files (read-only)
data/preprocess/<dataset>/<dataset>__prep-vNN__<ts>/   train/test parquet, manifest.json, copy of the pipeline
results/<dataset>/runs/<ts>__<dataset>__<scenario>__<strategy>__<model>__<prep-vNN>__s<seed>/
    __results.json   raw pyCLAD output: metric matrices + summary metrics, time, cost
    __config.json, __env.json (versions + git commit), __run.log, __heatmap_<metric>.png/.pdf
results/<dataset>/runs_index.jsonl                one line per run (append-only)
results/<dataset>/evaluations/<ts>__eval__<scope>__<metric>__<cost>/
    __comparison.csv/.json   mean/std/n per configuration (all summary metrics)
    __table_main.csv/.md     paper table: ContinualAverage, DiagonalAverage, PastConceptsAverage, BWT, FM, cost
    __paired_vs_naive.csv    seed-paired difference vs Naive with 95% CI
    __cellwise_vs_naive.csv  cells better / equal / worse than Naive (diagonal, past, all)
    __tradeoff.csv + fig_tradeoff_*      retention vs cost, 0 = Naive, 1 = Cumulative
    fig_heatmaps_*, fig_heatmap-diff-vs-naive_*, fig_diagonal_*, fig_cost-vs-perf_*
    __summary.md
research/proposals/  research/backlog/  research/CHANGELOG.md  research/research_log.md
```

Do not delete or edit anything in `data/raw/` or `results/*/runs/`. Every run records `lfd_version` and
`git_commit`, so each number can be traced to the code that produced it.

## 5. Adding a strategy (or model, or metric)

pyCLAD (`src/pyclad/`, `tests/`, `docs/`, `pyproject.toml`) is **upstream code: don't edit it.** Extend it from `lfd/`:

1. **Strategy.** Write a builder in `lfd/registry.py` and add it to `STRATEGIES` (and `STRATEGY_CLASSES` so
   incompatible scenarios are skipped):
   ```python
   STRATEGIES["MyStrategy"] = lambda mf, **p: MyStrategy(mf(), alpha=float(p.get("alpha", 0.5)))
   ```
   `mf()` returns a fresh model wrapped in `CostTrackingModel`. Always build models through it, otherwise the cost
   is not counted. A strategy that is not a pyCLAD class goes in its own module under `lfd/` and subclasses
   `pyclad.strategies.strategy.ConceptIncrementalStrategy` (implement `learn`, `predict`, `name`).
   For torch strategies (EWC, LwF, A-GEM …) follow `examples/strategies/*.py` in pyCLAD.
   Hyper-parameters must not be tuned on test concepts (open question BL-0009).
2. **Model.** Add to `MODELS` (and to `SEEDABLE` if it accepts `random_state`).
3. **Summary metric.** Subclass `pyclad.metrics.continual.concepts_metric.SummarizedMetric` in `lfd/metrics.py` and
   append it to `SUMMARY_METRICS`. If old runs should get it too, compute it post-hoc in `lfd/evaluate.py` from the
   saved `metric_matrix` (as for `DiagonalAverage`).
4. **Behaviour change** (e.g. a different sampling of the stream). Make it an opt-in feature in
   `lfd/features/<name>.py`, off by default, so old configs keep giving the same numbers.
5. Add tests in `tests_lfd/` and a line under `[Unreleased]` in `research/CHANGELOG.md`. Run `pytest tests_lfd -q`.

## 6. Good scientific practice

- Always include `Naive` and `Cumulative`. Use at least 3 seeds, and 5 for the paper. Report mean ± std and n.
- For fraud (rare anomalies) lead with **AP**, not ROC-AUC.
- `MSTE` in `concept_aware` knows the concept id at test time (an oracle). Say so whenever it appears.
- Report failed runs and negative results; never drop runs to make a figure look better.
- Log every decision in `research/research_log.md` (append-only).
