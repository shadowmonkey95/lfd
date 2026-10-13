---
id: BL-0008
title: paper-main-table-and-aggregated-json
status: done
type: improvement
scope: core-lfd
priority: P2
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-13T00:03:10
branch: bl-0005-diagonal-metrics-heatmaps-docs
commits: 6fe38f3
feature_flag: -
changelog: Unreleased
---
## Request
PROF: "usually in the paper we do a big table with aggregated metrics, and display a few selected heatmaps and upload the rest of them on GitHub."
PROF: "the aggregated ROC-AUC ... by default is saved in pyCLAD as JSON file. But I don't remember seeing it in the zip file of your results."
Researcher (note 4): main table = ContinualAverage, DiagonalAverage, PastConceptsAverage, BWT, Forgetting, cost; AP and ROC-AUC.

## Interpretation
The report-writer's main table gets the new columns for both base metrics. To make the aggregated numbers travel with the figures, each evaluation folder also writes `__table_main.csv/.md` (mean ± std, n) and `__comparison.json`, so a zip of the eval folder carries the aggregated metrics, not only heatmaps.

## Acceptance criteria
- `lfd.evaluate` writes `__table_main.csv` + `__table_main.md` with columns ContinualAverage, DiagonalAverage, PastConceptsAverage, BackwardTransfer, ForgettingMeasureStrict (mean ± std), cost relative to Cumulative, n_seeds; and `__comparison.json`.
- `.claude/agents/report-writer.md` builds the paper table from those columns, for AP first then ROC-AUC (gitignored, not committed).

## Scope decision
core-lfd (reporting tooling, additive).

## Plan
`lfd/evaluate.py`, `.claude/agents/report-writer.md`.

## Open questions
None.
