# lfd — Lifelong Fraud Detection on pyCLAD

Research project: show that continual anomaly-detection strategies **avoid forgetting while saving cost** on
time-ordered fraud data (IEEE-CIS). Built on top of [pyCLAD](https://github.com/lifelonglab/pyCLAD)
(`README.md` in this repo is pyCLAD's own and is left untouched).

```
src/pyclad/ tests/ examples/ docs/ pyproject.toml README.md   ← upstream pyCLAD — do not edit (keeps upgrades clean)
lfd/                     our code: naming, runner, evaluation, cost, approval gate, feature flags, backlog
  streams/time_windows.py   time-based concept streams (port of the legacy "time_based" scenario)
  datasets/ieee_ead.py      artgor/EAD feature engineering for IEEE-CIS
  features/                 opt-in switches (python -m lfd.features)
tests_lfd/               our tests                    pipelines/<dataset>/prep_vNN.py   preprocessing versions
configs/experiments/     one YAML = one sweep         references/notebooks/            reference notebooks
research/                proposals, backlog, CHANGELOG, research log, UPSTREAM_PATCHES.md
data/raw, data/preprocess, results/, research/meetings/, .claude/   ← local only (gitignored)
```
Old project: tag `legacy-v1` and branches `legacy/*` (read-only).

---

## 1. Setup (once per machine)

```bash
git clone https://github.com/shadowmonkey95/lfd.git && cd lfd
git remote add upstream https://github.com/lifelonglab/pyCLAD.git     # for future pyCLAD upgrades

python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e .                     # pyCLAD (editable, from src/)
pip install -r requirements-lfd.txt
pip install torch                    # only for AE / VAE models

pytest tests_lfd -q                  # should pass
```

Put the Kaggle IEEE files, unchanged, in `data/raw/ieee-fraud-detection/`
(`train_transaction.csv`, `train_identity.csv`).

### Claude configuration (local, not in git)
`.claude/` holds the agents, slash commands, permission rules and `CLAUDE.md`. It is gitignored, so it never shows
up on GitHub. Restore it on a new machine by unzipping `lfd-claude-config.zip` into the repo root
(you should then have `.claude/CLAUDE.md`, `.claude/agents/`, `.claude/commands/`, `.claude/settings.json`).

Recommended backup — make `.claude/` its own private repo (git ignores it from the outer repo, so they don't mix):
```bash
cd .claude && git init && git add -A && git commit -m "claude config"
git remote add origin https://github.com/shadowmonkey95/lfd-claude.git   # create it as PRIVATE first
git push -u origin main && cd ..
```
On another machine: `git clone https://github.com/shadowmonkey95/lfd-claude.git .claude` inside the repo.

---

## 2. Run without Claude (plain commands)

```bash
# 1) preprocessing needs an approved proposal (the approval gate)
python -m lfd.proposals new --kind data --dataset ieee-fraud-detection --title "legacy time-based v01" --body-file RESEARCH.md
python -m lfd.proposals approve research/proposals/ieee-fraud-detection/<file printed above>
python -m lfd.prep --raw ieee-fraud-detection --pipeline pipelines/ieee-fraud-detection/prep_v01.py --proposal <same file>

# 2) train: a quick pilot first, then the grid that reproduces the legacy main.py
python -m lfd.run --config configs/experiments/ieee-fraud-detection/pilot-iforest-1seed.yaml
python -m lfd.run --config configs/experiments/ieee-fraud-detection/legacy-timebased-grid.yaml --dry-run
python -m lfd.run --config configs/experiments/ieee-fraud-detection/legacy-timebased-grid.yaml

# 3) evaluate (tables, heatmaps, cost-vs-retention trade-off)
python -m lfd.evaluate --dataset ieee-fraud-detection --metric AP
python -m lfd.status
```
Every output is timestamped and named after dataset, scenario, strategy (+params), model (+params), data version,
feature switches and seed, e.g.
`results/ieee-fraud-detection/runs/20261012-140501__ieee-fraud-detection__incremental__ReplayEnhanced-buf1000__AE-ep20-h32-24-16__prep-v01__s0/`.
Nothing is ever overwritten.

### The time-based stream (what prep_v01 does)
1. artgor/EAD features, sorted by `TransactionDT`.
2. Normal transactions cut into 8 consecutive equal-count windows (= concepts).
3. Each window gets the frauds inside its time span (capped at `n_frauds // 8`; nearest-in-time if none).
4. First 60% of a window's normals → train (normal only); remaining normals + the window's frauds → test.
5. `TransactionDT` removed from the features.

This is bit-for-bit the legacy `time_based` scenario. Known caveats are in `research/BACKLOG.md` (BL-0001..0003);
fixes are options of `TimeStreamConfig` (e.g. `anomaly_span="test_period"`) and go into `prep_v02` after approval.

---

## 3. Working with Claude Code

Start it from the repo root (venv active): `claude`. Then:

| You type | What happens |
|---|---|
| `/status` | where everything stands |
| `/process-data ieee-fraud-detection <kaggle-notebook-url>` | profiles the data, proposes a new prep version with questions — processes nothing |
| answer, then `/approve latest` | permission prompt → preprocesses into a new versioned folder |
| `/train ieee-fraud-detection pilot first, then the legacy grid with 5 seeds` | writes a sweep config, dry-runs, runs |
| `/evaluate ieee-fraud-detection last` | findings, heatmaps, trade-off figure, improvement proposals (not run) |
| `/review last` | independent audit (leakage, statistics, claims) |
| `/meeting <paste a conversation with your professor>` | saved verbatim → backlog items → implemented (generic → code, specific → opt-in switch) |
| `/backlog`, `/implement BL-0002`, `/release` | backlog board, build one item, cut a version |
| `/sync-upstream` | brings in the newest pyCLAD on a test branch (see §4) |

Good habits
- Let agents propose; you approve. Approvals and `git push` always ask you first.
- One question per message works best; paste errors in full.
- Before thesis numbers: `/release`, then run sweeps — every run records `lfd_version` and the git commit.
- `research/research_log.md` is the paper trail; ask Claude to add an entry after each decision.

Git with Claude: the framework-engineer works on branches `bl-XXXX-<topic>`, merges into `main` only when tests pass,
and never pushes. You push: `git push origin main`.

---

## 4. Upgrading pyCLAD

`main` is built on upstream pyCLAD's history, so an upgrade is a normal git merge. Because we don't edit upstream
files (see `research/UPSTREAM_PATCHES.md`), it normally merges without conflicts.

With Claude: `/sync-upstream`, read its report, then tell it to merge.

By hand:
```bash
git switch main && git pull                       # clean working tree first
git fetch upstream --tags
git log --oneline main..upstream/main             # what's new upstream
git switch -c sync/upstream-$(date +%Y%m%d)
git merge --no-ff upstream/main                   # conflicts? only expected in .gitignore (keep both parts)
pip install -e .
pytest tests_lfd -q && pytest tests -q -x
python -m lfd.run --config configs/experiments/synthetic-demo/baseline-grid.yaml --dry-run
git switch main && git merge --no-ff sync/upstream-YYYYMMDD && git push origin main
```
If a test breaks because pyCLAD renamed something, fix it in `lfd/` (e.g. `lfd/run.py`, `lfd/registry.py`), not in
`src/pyclad/`. Add a line to `research/CHANGELOG.md` ("Synced with pyCLAD vX.Y") — results produced before and after
an upgrade carry different git commits, so they stay distinguishable.

To pin a released pyCLAD instead of upstream `main`: `git merge v0.4.0` (upstream tags) instead of `upstream/main`.
