---
id: BL-0006
title: evaluate-posthoc-knowledge-accumulation
status: done
type: feature
scope: core-lfd
priority: P1
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-13T00:03:10
branch: bl-0005-diagonal-metrics-heatmaps-docs
commits: 6fe38f3
feature_flag: -
changelog: Unreleased
---
## Request
ME: "very rarely Replay decreases a single score, and in quite a few of the cells Replay increases the scores."
Researcher (note 2): compute the new metrics post-hoc in lfd.evaluate; cell-wise better/equal/worse vs Naive; per-concept diagonal figure.

## Interpretation
Old runs were trained without the new metrics, so lfd.evaluate recomputes DiagonalAverage and PastConceptsAverage from the saved `metric_matrix` of every run's `__results.json` (no retraining). The researcher's cell-count observation is quantified seed-paired.

## Acceptance criteria
- `__comparison.csv`, `__summary.md` and `__paired_vs_naive.csv` contain `<M>__DiagonalAverage` and `<M>__PastConceptsAverage` for runs made before 0.3.0.
- New `__cellwise_vs_naive.csv`: per scenario/model/setting/config and region (all = diagonal+lower, diagonal, past), the number of cells better / equal (|diff| <= 0.005) / worse than Naive, per seed-pair then mean ± std over seeds, n.
- New figure `__fig_diagonal_<scen>_<model>.png/.pdf`: per-concept M[i][i], mean ± std over seeds, one line per config.
- Test on a synthetic runs index + results.json files in tmp dir.

## Scope decision
core-lfd: evaluation tooling; adds columns/files, never changes an existing number.

## Plan
`lfd/evaluate.py`, `tests_lfd/test_evaluate.py`.

## Open questions
None. (Tolerance 0.005 is a CLI option `--cell-tol`.)
