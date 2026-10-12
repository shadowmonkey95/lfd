---
id: BL-0002
title: test-anomalies-from-test-period
status: new
type: research-idea
scope: data
priority: P1
source: 
created: 2026-10-12T11:28:58
updated: 2026-10-12T11:29:05
branch: 
commits: 
feature_flag: 
changelog: 
---
## Request
From the 0.2.0 rebuild review of the legacy time-based code.

## Interpretation
Legacy takes the EARLIEST frauds of each window (cap applied in time order), i.e. frauds from the training period, while test normals come from the last 40% of the window. Test anomalies and normals then also differ by time. TimeStreamConfig(anomaly_span='test_period') already implements the fix; needs a prep_v02 data proposal and a comparison sweep vs prep-v01.
