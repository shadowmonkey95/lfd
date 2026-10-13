# Changelog

All notable changes to this research codebase (pyCLAD core in `src/pyclad`, the harness in `lfd/`,
opt-in features in `lfd/features/`). Format: [Keep a Changelog](https://keepachangelog.com), versions = `lfd.LFD_VERSION`.
Maintained by the framework-engineer agent. Every entry names its backlog item, scope tag and how to use it.

Scope tags: **[core-pyclad]** on by default in pyCLAD · **[core-lfd]** harness, on by default ·
**[opt-in]** off unless enabled in a sweep (`features:` or `--feature`) · **[behaviour-change]** changes numbers of existing configs.

## [Unreleased]
### Added
- [core-lfd] `DiagonalAverage` (mean of M[i][i], the just-learned concept; reuses pyCLAD's
  `ScheduleAwareNewTaskAcquisition` with the identity schedule) and `PastConceptsAverage` (mean of M[i][j], j<i)
  in `lfd/metrics.py`, registered in `SUMMARY_METRICS`. `ContinualAverage` unchanged. New runs store both in
  `__results.json` and `runs_index.jsonl`. (BL-0005)
  Use: automatic; columns `AP__DiagonalAverage`, `AP__PastConceptsAverage`, ...
- [core-lfd] `lfd.evaluate` recomputes both metrics post-hoc from each run's saved `metric_matrix` (older runs
  included, no retraining) and adds them to `__comparison.csv`, `__summary.md` and `__paired_vs_naive.csv`. New
  `__cellwise_vs_naive.csv` (seed-paired better / equal / worse cell counts vs Naive for regions all/diagonal/past)
  and `__fig_diagonal_<scen>_<model>` (per-concept M[i][i], mean ± std). (BL-0006)
  Use: `python -m lfd.evaluate --dataset ieee-fraud-detection --sweep <id> --metric AP [--cell-tol 0.005]`
- [core-lfd] `__table_main.csv/.md` (ContinualAverage, DiagonalAverage, PastConceptsAverage, BWT,
  ForgettingMeasureStrict as mean ± std, cost relative to Cumulative, n) and `__comparison.json` in every
  evaluation folder; report-writer builds the paper table from them for AP and ROC-AUC. (BL-0008)
- [core-lfd] `research/docs/SCENARIOS_AND_DATASETS.md` and `research/docs/RUNNING_EXPERIMENTS.md`. (BL-0010)
### Changed
- [core-lfd] **Default changed at the researcher's explicit request (meeting 2026-10-12, item 3):** heatmaps
  (per-run `__heatmap_*` and evaluation `fig_heatmaps_*` / `fig_heatmap-diff-vs-naive_*`) hide cells above the
  diagonal. Plots only; no metric, CSV or run id changes. (BL-0007)
  Use (show them again): sweep YAML `heatmap_show_upper_diagonal: true` or `python -m lfd.run ... --heatmap-show-upper-diagonal`;
  `python -m lfd.evaluate ... --show-upper-diagonal`.
### Fixed
- [core-lfd] `lfd.evaluate` heatmap grids looked up the results block as `concept_metric_callback_<CLI name>`,
  so `--metric F1` (block `F1-Score`) silently produced no heatmaps; now resolved via the registry. No numbers change.

## [0.2.0] - 2026-10-12
Rebuilt the project on the latest upstream pyCLAD (v0.3.0+, commit 9de39e1). Old project preserved as tag `legacy-v1`
and branches `legacy/*`.
### Added
- [core-lfd] `lfd/streams/time_windows.py` — generic time-based concept streams (`build_time_concepts`,
  `TimeStreamConfig`). Defaults reproduce legacy `time_based` scenario exactly (bit-identical concepts verified on
  IEEE-shaped data, incl. the empty-window fallback). Opt-in alternatives: `window=equal_time`,
  `anomaly_span=test_period`, `anomaly_pick=spread`, explicit `anomaly_cap`, `tail=include|overflow`.
- [core-lfd] `lfd/datasets/ieee_ead.py` — artgor/EAD feature engineering for IEEE-CIS (port of
  `ead_preprocess_keepTransactionDT.py`, in memory, parameterised).
- [core-lfd] `pipelines/ieee-fraud-detection/prep_v01.py` — legacy-faithful IEEE time-based stream (8 concepts).
- [core-lfd] Models `AE`, `VAE` (pyOD, torch loaded lazily) in `lfd/registry.py`.
- [core-lfd] Configs `ieee-fraud-detection/legacy-timebased-grid.yaml` (reproduces legacy main.py) and
  `pilot-iforest-1seed.yaml`.
- [core-lfd] Preprocessing manifests record `lfd_version` and `git_commit`.
### Not ported (on purpose)
- Custom `PrAuc` metric — identical to pyCLAD's built-in `AP` (`average_precision_score` on anomaly scores).
- Clustering scenarios (`random_anomalies`, `clustered_*`) — still in `legacy-v1`; see BL-0004.
- Exploration notebooks, tmp files, CSV writers, `pipeline.py` UNSW/NSL-KDD demo.
### Changed
- [core-lfd] Package `lab` → `lfd`; research records moved to `research/`; Claude config to `.claude/` (gitignored).
- Legacy `ConceptMetricCallback(metrics=...)` is `summarized_metrics=` in current pyCLAD — handled by `lfd.run`.

## [0.1.0] - 2026-10-12
### Added
- [core-lfd] Research harness: naming/never-overwrite, approval gate, versioned preprocessing, sweep runner,
  cost accounting (`samples_fitted_total`), evaluation with trade-off figures. (BL-0000)
- [core-lfd] Opt-in feature system (`lfd/features/`, `--feature NAME:k=v`), feature tag in run ids. (BL-0000)
- [opt-in] `concept_order` — reverse / shuffle / custom concept stream order.
  Use: `--feature concept_order:order=shuffle,seed=1`. (BL-0000)
- [core-lfd] Backlog + meeting notes tooling (`lfd/backlog.py`). (BL-0000)
