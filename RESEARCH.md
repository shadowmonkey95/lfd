# pyCLAD research kit — agent system for "avoid forgetting while saving cost"

Drop-in Claude Code setup for a clone of https://github.com/lifelonglab/pyCLAD.

## Install (once)
```bash
git clone https://github.com/lifelonglab/pyCLAD.git && cd pyCLAD
# copy EVERYTHING from this kit into the repo root (.claude/CLAUDE.md, .claude/, lfd/, pipelines/, configs/, data/, ...)
pip install -e .                       # pyCLAD itself
pip install -r requirements-lfd.txt
pytest tests_lfd -q
cat gitignore-additions.txt >> .gitignore   # keep raw data and results out of git
```

## Smoke test (2 minutes, no real data)
```bash
python -m lfd.proposals new --kind data --dataset synthetic-demo --title "smoke test" --body-file RESEARCH.md
python -m lfd.proposals approve research/proposals/synthetic-demo/<the file just printed>
python -m lfd.prep --raw synthetic-demo --pipeline pipelines/synthetic-demo/prep_v01.py --proposal <same file>
python -m lfd.run --config configs/experiments/synthetic-demo/baseline-grid.yaml
python -m lfd.evaluate --dataset synthetic-demo --metric AP
python -m lfd.status
```

## Working with Claude Code (from the repo root: `claude`)
| You type | What happens |
|---|---|
| `/status` | where everything stands, what's pending |
| `/process-data ieee-fraud https://www.kaggle.com/code/...` | data-processor profiles `data/raw/ieee-fraud`, reads the notebook, writes `pipelines/ieee-fraud/prep_v01.py` + a proposal with questions. Nothing is processed yet. |
| answer its questions, then `/approve latest` | you get a permission prompt; on yes it writes `data/preprocess/ieee-fraud/ieee-fraud__prep-v01__<ts>/` |
| `/train ieee-fraud baseline grid, IForest + ECOD, 5 seeds` | experiment-runner writes a sweep config and runs it |
| `/evaluate ieee-fraud last` | evaluator: tables, heatmaps, trade-off figure, FINDINGS.md, improvement proposals (not run) |
| `/review last` | research-reviewer independently audits leakage, stats, claims → REVIEW.md |
| `/approve <proposal>` / `/reject <proposal> reason` | you decide which improvements run |
| `/report <eval folders>` | report-writer: LaTeX/MD tables, figures, results + methods text |
| `/meeting <paste your conversation with the professor>` | framework-engineer saves it verbatim, creates backlog items, implements generic ones in the code and specific ones as opt-in feature flags, updates research/BACKLOG.md + research/CHANGELOG.md (add "plan only" to skip implementing) |
| `/backlog`, `/implement BL-0003`, `/release` | see the board, build one item, cut a version |

### Opt-in features ("an arg I add before training")
Ideas that would change results are built as switches, off by default:
```yaml
features:
  - {name: concept_order, params: {order: shuffle, seed: 1}}
```
or `python -m lfd.run --config ... --feature concept_order:order=shuffle,seed=1`.
`python -m lfd.features` lists what exists. The feature appears in the run name (`__ft-conceptorder-shuffle__`) and the
evaluator compares each feature setting against its own Naive/Cumulative baselines.

## Guarantees
- **Never overwritten**: every folder is `<timestamp>__...` and created with `exist_ok=False`
  (same-second collision → `__r2`). Files inside repeat the folder name.
- **Names say everything**: `20261012-001420__ieee-fraud__incremental__ReplayEnhanced-buf200__IsolationForest-n100__prep-v02__s3`
- **Human gate**: preprocessing and agent-suggested experiments refuse to run without an approved proposal;
  approving always prompts you. `data/raw`, `data/preprocess` and run folders are write-protected for Claude.
- **Traceable**: each run stores its resolved config, environment (package versions, git commit), log and raw
  pyCLAD JSON; each dataset version stores the exact pipeline code + the proposal that approved it.

## How the claim is measured
- Retention: pyCLAD's ContinualAverage, BackwardTransfer, ForgettingMeasureStrict (on AP and ROC-AUC).
- Cost: `samples_fitted_total` (hardware-independent, from `lfd/cost.py`), plus fit seconds, train time,
  optional peak memory (`track_memory: true`) and energy (`energy: true`).
- Headline figure `fig_tradeoff_*`: both axes normalised so Naive = 0 and Cumulative = 1.
  Points above the diagonal retain more than their share of cost — that's the evidence for the hypothesis.
- `paired_vs_naive.csv`: seed-paired differences with 95% CIs.

## Extending
Add strategies/models in `lfd/registry.py` (pyOD-based ones are one line; torch strategies such as EWC, LwF,
A-GEM, DER++, VLAD need a builder like pyCLAD's `examples/strategies/*.py`).
