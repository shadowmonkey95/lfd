---
id: BL-0003
title: fallback-borrows-shared-future-frauds
status: new
type: research-idea
scope: data
priority: P2
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
When a window has no frauds, legacy fallback takes frauds around the window start, which can come from later windows and be shared by several concepts (seen in verification: Concept_0 and Concept_1 got the same frauds). Proposal: disable fallback (nearest_fallback=False) and drop/merge fraud-free windows, or report them; decide with the researcher.
