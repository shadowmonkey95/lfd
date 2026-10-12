# Changelog

All notable changes to this research codebase (pyCLAD core in `src/pyclad`, the harness in `lfd/`,
opt-in features in `lfd/features/`). Format: [Keep a Changelog](https://keepachangelog.com), versions = `lfd.LFD_VERSION`.
Maintained by the framework-engineer agent. Every entry names its backlog item, scope tag and how to use it.

Scope tags: **[core-pyclad]** on by default in pyCLAD · **[core-lfd]** harness, on by default ·
**[opt-in]** off unless enabled in a sweep (`features:` or `--feature`) · **[behaviour-change]** changes numbers of existing configs.

## [Unreleased]

## [0.1.0] - 2026-10-12
### Added
- [core-lfd] Research harness: naming/never-overwrite, approval gate, versioned preprocessing, sweep runner,
  cost accounting (`samples_fitted_total`), evaluation with trade-off figures. (BL-0000)
- [core-lfd] Opt-in feature system (`lfd/features/`, `--feature NAME:k=v`), feature tag in run ids. (BL-0000)
- [opt-in] `concept_order` — reverse / shuffle / custom concept stream order.
  Use: `--feature concept_order:order=shuffle,seed=1`. (BL-0000)
- [core-lfd] Backlog + meeting notes tooling (`lfd/backlog.py`). (BL-0000)
