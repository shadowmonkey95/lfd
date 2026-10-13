---
id: BL-0010
title: scenario-and-running-docs
status: in-progress
type: chore
scope: core-lfd
priority: P2
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-12T23:55:38
branch: bl-0005-diagonal-metrics-heatmaps-docs
commits: 
feature_flag: 
changelog: 
---
## Request
PROF: "if you can distill an .md file that explains how the scenarios / datasets were constructed, I can write about it in the paper. And maybe it is a good idea for next students who want to work on it, knowing how experiments can be run, etc."

## Interpretation
Two plain-language docs in research/docs/ (not upstream docs/): SCENARIOS_AND_DATASETS.md (IEEE time-based stream from prep_v01: preprocessing, windowing, train/test split, anomaly assignment, caveats BL-0001..0003) and RUNNING_EXPERIMENTS.md (uv setup, commands, folders/naming, adding a strategy).

## Acceptance criteria
- Every statement about the stream traceable to `pipelines/ieee-fraud-detection/prep_v01.py`, `lfd/streams/time_windows.py`, `lfd/datasets/ieee_ead.py`.
- Metric definitions (ContinualAverage vs DiagonalAverage vs PastConceptsAverage) explained.

## Scope decision
core-lfd (docs, chore).

## Plan
`research/docs/SCENARIOS_AND_DATASETS.md`, `research/docs/RUNNING_EXPERIMENTS.md`.

## Open questions
Concrete numbers (rows per concept, fraud counts) need a manifest; none exists yet because prep-v01 has not been run in this repo. Docs say so and point to the manifest fields.
