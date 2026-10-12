---
id: BL-0001
title: causal-preprocessing-fit
status: new
type: research-idea
scope: data
priority: P1
source: 
created: 2026-10-12T11:28:58
updated: 2026-10-12T11:28:58
branch: 
commits: 
feature_flag: 
changelog: 
---
## Request
From the 0.2.0 rebuild review of the legacy time-based code.

## Interpretation
Scaler, LabelEncoders, medians and TransactionAmt/id_02/D15 group statistics are fit on ALL rows (incl. future concepts). In a continual stream this leaks the future into early concepts. Proposal: prep_v02 fits them on concept 0's training rows only (or expanding past-only window) and applies forward. Acceptance: no statistic uses rows with TransactionDT later than the concept being transformed; manifest documents fit scope.
