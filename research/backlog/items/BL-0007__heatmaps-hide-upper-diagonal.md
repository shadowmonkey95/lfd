---
id: BL-0007
title: heatmaps-hide-upper-diagonal
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
changelog: Unreleased (researcher-requested default change)
---
## Request
PROF: "it would be great if in heatmaps we can hide the entries above the diagonal."
Researcher (note): "current heatmaps show the upper triangle ... hide entries above the diagonal by default ... with a config/CLI option to show them."

## Interpretation
Upper-triangle cells (evaluating on concepts not yet learned) are hidden in per-run heatmaps (lfd.run) and in lfd.evaluate heatmap grids and diff-vs-Naive heatmaps.

## Acceptance criteria
- Default: hidden. Show again with sweep YAML `heatmap_show_upper_diagonal: true` or `python -m lfd.run ... --heatmap-show-upper-diagonal`, and `python -m lfd.evaluate ... --show-upper-diagonal`.
- No metric value, CSV or run id changes.

## Scope decision
core-lfd. **Default change explicitly requested by the researcher** in the 2026-10-12 meeting notes (item 3); plots only, numbers unchanged, so no proposal needed.

## Plan
`lfd/run.py`, `lfd/evaluate.py`, tests.

## Open questions
None.

## Resolution
Implemented as a **researcher-requested default change** (meeting 2026-10-12, researcher notes item 3: "hide entries above the diagonal by default"). Plots only; no numbers change. Recorded in CHANGELOG [Unreleased] → Changed.
