---
id: BL-0009
title: agem-and-hyperparameter-validation
status: needs-clarification
type: feature
scope: core-lfd
priority: P2
source: research/meetings/20261012-235445__meeting__prof-sync-knowledge-accumulation-diagonal-ROC-AUC-heatmaps-A.md
created: 2026-10-12T23:55:38
updated: 2026-10-12T23:55:38
branch: 
commits: 
feature_flag: 
changelog: 
---
## Request
PROF: "Another method, AGEM also, shouldn't there. The only issue with some of these methods is that they have one or two hyperparameters that should be tweaked, ideally with a validation set."
Researcher (note 5): create item, needs-clarification, do not implement.

## Interpretation
Add A-GEM (and later other hyper-parameterised strategies, e.g. EWC/LwF) to `lfd/registry.py`, with a hyper-parameter selection protocol that never touches test data.

## Acceptance criteria (draft, after clarification)
- A-GEM registered (torch backbone as in pyCLAD examples), cost tracked.
- Hyper-parameter tuning protocol implemented as an opt-in feature / separate tuning sweep; test concepts never used for selection; selected values recorded in the run config.

## Scope decision
Strategy itself: core-lfd (additive). Validation-split protocol: opt-in-feature or data (it changes what data each concept trains on) — decided after the answer below.

## Plan
Not started (blocked on clarification).

## Open questions (for the professor)
Where should validation labels come from?
(a) tune on the first 1-2 concepts and report only on the remaining concepts, or
(b) a labelled validation slice carved from each concept's data (never the test set) — this needs a new prep version since training data shrinks.
Also: which hyper-parameters and search ranges for A-GEM (memory size, patterns per step)?
