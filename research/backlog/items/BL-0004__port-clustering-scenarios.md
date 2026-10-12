---
id: BL-0004
title: port-clustering-scenarios
status: new
type: improvement
scope: core-lfd
priority: P3
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
Legacy also had random_anomalies / clustered_with_random_assignment / clustered_with_closest_assignment / all_anomalies (from lifelonglab/lifelong-anomaly-detection-scenarios). Port as lfd/streams/clustered.py only if the thesis needs non-temporal baselines.
