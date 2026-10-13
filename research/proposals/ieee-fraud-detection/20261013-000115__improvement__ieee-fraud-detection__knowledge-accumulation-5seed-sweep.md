---
id: 20261013-000115__improvement__ieee-fraud-detection__knowledge-accumulation-5seed-sweep
kind: improvement
dataset: ieee-fraud-detection
status: pending
approved_at: 
title: knowledge-accumulation-5seed-sweep
---
# Knowledge-accumulation sweep (BL-0011)

## What changes
No code. A new sweep config (copy of `legacy-timebased-grid.yaml`, new name e.g. `knowledge-accumulation-5seed.yaml`)
with seeds [0, 1, 2, 3, 4]; same prep-v01 data, scenario `concept_incremental`, strategies Naive / Cumulative /
ReplayEnhanced(buffer_size=1000), models IsolationForest, ECOD, COPOD, AE; metrics [ROC-AUC, AP].
Evaluate with `python -m lfd.evaluate ... --metric AP` and `--metric ROC-AUC` (lfd >= 0.3.0 adds DiagonalAverage,
PastConceptsAverage, cellwise_vs_naive.csv and fig_diagonal_*).

## Why
Meeting 2026-10-12 (research/meetings/20261012-235445__meeting__...): the researcher observed that Replay rarely
lowers a single heatmap cell vs Naive and often raises it ("knowledge accumulation"). The professor asked to check
it on aggregated metrics and on the diagonal separately. 5 seeds are required for paper numbers.

## Expected effect
Paired-vs-Naive CIs for DiagonalAverage and PastConceptsAverage, plus better/equal/worse cell counts. Hypothesis:
ReplayEnhanced > Naive on PastConceptsAverage (retention) and >= Naive on DiagonalAverage, at cost well below
Cumulative. A null result on the diagonal is also a finding and must be reported.

## Risks
- prep-v01 caveats BL-0001..0003 (leakage via global scaler, test frauds from training period, shared fallback
  frauds) apply to every number; state them or wait for prep_v02.
- data/preprocess/ieee-fraud-detection is currently empty: prep-v01 needs its own approved data proposal first.
- AE runs on ~340k rows for Cumulative are slow; run a 1-seed pilot first.

## Questions for the researcher
1. Run on prep-v01 now (comparable with legacy results) or wait for prep_v02 (BL-0001/0002 fixes)?
2. Is 3 seeds (existing legacy-timebased-grid) enough as a first look, with 5 seeds only for the thesis sweep?
3. The meeting mentions "7 concepts"; prep_v01 builds 8. Which stream did the shared heatmaps come from?
