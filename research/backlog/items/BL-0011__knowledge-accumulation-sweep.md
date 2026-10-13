---
id: BL-0011
title: knowledge-accumulation-sweep
status: blocked
type: research-idea
scope: experiment
priority: P2
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-13T00:03:10
branch: proposal research/proposals/ieee-fraud-detection/20261013-000115__improvement__ieee-fraud-detection__knowledge-accumulation-5seed-sweep.md
commits: 
feature_flag: 
changelog: 
---
## Request
ME: "maybe this is enough to support the idea of CL in time-based scenarios and an upgrade over online naive learning?"
PROF: "To check this we would have to look at the aggregated ROC-AUC"; "we will compare Naive vs. Replay and other methods."

## Interpretation
Research hypothesis "knowledge accumulation": in the time-based IEEE stream Replay beats Naive on the diagonal (just-learned concept) and on past concepts, at lower cost than Cumulative. Test it with the existing baseline grid (Naive / Cumulative / ReplayEnhanced) evaluated with the new metrics (BL-0005/0006); for the paper, 5 seeds.

## Acceptance criteria
- Evaluation of the sweep for AP and ROC-AUC with DiagonalAverage/PastConceptsAverage paired-vs-Naive CIs and cell-wise counts; reviewed by research-reviewer.

## Scope decision
experiment — no code. Routed to experiment-runner via an improvement proposal (5-seed variant of legacy-timebased-grid). Blocked in practice on prep-v01 data existing (data/preprocess/ieee-fraud-detection is empty; no IEEE sweep has run in this repo).

## Plan
Proposal in research/proposals/ieee-fraud-detection/.

## Open questions
Is the 3-seed `legacy-timebased-grid` enough as a first look, with 5 seeds only for the thesis sweep?

## Status
Blocked on proposal research/proposals/ieee-fraud-detection/20261013-000115__improvement__ieee-fraud-detection__knowledge-accumulation-5seed-sweep.md (pending) and on prep-v01 data existing (needs its own approved data proposal).
