---
id: BL-0005
title: diagonal-and-past-concepts-metrics
status: in-progress
type: feature
scope: core-lfd
priority: P1
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-12T23:55:38
branch: bl-0005-diagonal-metrics-heatmaps-docs
commits: 
feature_flag: 
changelog: 
---
## Request
PROF: "we should also track entries on the main diagonal only. It is possible to do this by adding one function in the metrics file."
Researcher (note 1): add DiagonalAverage + PastConceptsAverage in lfd/, keep ContinualAverage unchanged.

## Interpretation
ContinualAverage = mean of diagonal + strictly-lower triangle (not "below the diagonal only" as the prof recalled). Split it:
- `DiagonalAverage` = mean_i M[i][i]: performance on the concept just learned (plasticity / knowledge accumulation).
  Implemented by delegating to pyCLAD's `ScheduleAwareNewTaskAcquisition` with the identity schedule `[0..N-1]`.
- `PastConceptsAverage` = mean over j<i of M[i][j]: retention on old concepts only.
NaN cells are ignored (same as NewTaskAcquisition). PastConceptsAverage on a 1-concept stream is NaN (there are no past concepts).

## Acceptance criteria
- Both are `SummarizedMetric`s in `lfd/metrics.py`, registered in `lfd/registry.py::SUMMARY_METRICS` (so new runs store them in `__results.json` and runs_index).
- Unit tests on a hand-made 3x3 matrix: diagonal mean, strict-lower mean, NaN handling, rejection of rectangular matrices, and that (N*Diag + P*Past)/(N+P) == ContinualAverage.
- ContinualAverage untouched; no edit under src/pyclad.

## Scope decision
core-lfd, additive: new metrics; no existing number changes (ContinualAverage etc. computed exactly as before). Not an upstream patch; could be an upstream PR candidate later (a `DiagonalAverage` convenience wrapper).

## Plan
`lfd/metrics.py` (new), `lfd/registry.py`, `tests_lfd/test_metrics.py`.

## Open questions
None.
