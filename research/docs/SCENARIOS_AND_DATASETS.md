# How the IEEE-CIS time-based continual stream is built

*Audience: the paper's "Dataset and scenario" section and future students. Plain language; every statement points to
the code that implements it. Backlog: BL-0010. Code version: lfd 0.2.0+ (prep_v01).*

Sources of truth, in order:
1. `pipelines/ieee-fraud-detection/prep_v01.py`: which steps run and every parameter (`PARAMS`).
2. `lfd/datasets/ieee_ead.py`: the feature engineering.
3. `lfd/streams/time_windows.py`: how rows become concepts (`TimeStreamConfig`, `build_time_concepts`).
4. The manifest of a processed dataset:
   `data/preprocess/ieee-fraud-detection/<folder>/<folder>__manifest.json`. It holds the exact per-concept counts
   and time ranges (`params.concepts`) and the list of dropped columns (`params.ead_report`).

> **Status note (2026-10-12).** prep_v01 has not been run in this repository yet, so no manifest exists. The counts
> below marked *expected* are derived from the public Kaggle file (590,540 transactions, 20,663 frauds) and the
> parameters. Replace them with the manifest values before they go into the paper.

prep_v01 is a **faithful port** of the original project's `time_based` scenario (tag `legacy-v1`, `src/main.py` with
`scenario_config="time_based"`, `concepts_no=8`, `size_per_concept=73817`). With these defaults it reproduces the old
concepts bit for bit. That makes old and new results comparable, but it also keeps the old caveats (see §5).

---

## 1. Raw data

Kaggle competition *IEEE-CIS Fraud Detection*, files `train_transaction.csv` and `train_identity.csv`, unchanged,
stored in `data/raw/ieee-fraud-detection/`. One row = one card transaction. `isFraud` is the label (1 = fraud).
`TransactionDT` is a time offset in seconds from an undisclosed reference date. We use it only to order
transactions in time. Only the labelled Kaggle *train* files are used; the Kaggle test files have no labels.

## 2. Feature preprocessing (`ieee_ead.preprocess`)

This follows the public Kaggle kernel "EDA and models" by artgor (copy in `references/notebooks/ieee-fraud-detection/`):

1. **Join.** Left-join the identity table onto the transactions by `TransactionID` (`use_identity=True`).
2. **Feature engineering.**
   - Ratios of `TransactionAmt`, `id_02` and `D15` to their mean and standard deviation within card/address groups
     (`card1`, `card4`, and for `D15` also `addr1`, `addr2`).
   - Each e-mail domain is split into up to three parts (e.g. `gmail`, `com`).
3. **Drop low-information columns.** A column is dropped if it is more than 90% missing, has one value in more than
   90% of rows, or has a single unique value (`low_info_threshold=0.9`). `isFraud` and `TransactionDT` are never
   dropped.
4. **Encode categoricals.** Categorical columns (card, address, device, e-mail, `M*`, `id_12…id_38`) are
   label-encoded as integers.
5. **Clean.** Sort by `TransactionDT`, drop `TransactionID`, turn ±inf into missing, fill missing values with the
   column median.
6. **Scale.** Standardise every feature to mean 0 and standard deviation 1 (`StandardScaler`), stored as float32.
7. Drop any row that still has a missing value. Normally there are none after step 5; the count is in the manifest
   as `rows_dropped_na`.

**Important:** steps 2, 4, 5 and 6 compute their statistics (group means, encoders, medians, scaler) on the **whole
dataset**, including transactions that belong to later concepts. See caveat BL-0001 in §5.

## 3. From a table to a stream of concepts (`build_time_concepts`)

A *concept* is one period of time. The model sees the concepts one after another, like a deployed fraud system
receiving new months of data.

1. **Separate and sort.** Split rows into normal (`isFraud=0`) and fraud (`isFraud=1`) rows, each sorted by
   `TransactionDT`.
2. **Windows of normal rows** (`window="equal_count"`, `n_concepts=8`). Cut the time-sorted normal rows into 8
   consecutive windows with the **same number of normal rows**. Each window holds
   `min(size_per_concept, n_normal // 8)` rows. With 569,877 normal rows that is `min(73817, 71234) = 71,234`
   (*expected*). Normal rows that don't fill a full window are dropped (`tail="drop"`; *expected* 5 rows).
   Busy periods therefore give shorter windows in calendar time. Equal-count windows are not equal-duration windows.
3. **Train/test split inside each window** (`train_ratio=0.6`). The earliest 60% of the window's normal rows (in
   time) form the **training set** of the concept: *expected* 42,740 rows. The training set contains **only normal
   transactions**, because the detectors are unsupervised one-class anomaly detectors. The remaining 40% of the
   normals (*expected* 28,494) go to the **test set**.
4. **Frauds for the test set** (`anomaly_span="window"`, `anomaly_pick="earliest"`, `anomaly_cap=-1`). Each concept's
   test set receives the frauds whose timestamp falls inside the window's time span (first to last normal row of the
   window). The number per concept is capped at `min(n_frauds // n_windows, 0.2 × size_per_concept)`, which is
   `min(20663 // 8, 14763) = 2,582` (*expected*). If a window has more frauds than the cap, the **earliest** ones are kept.
5. **Fallback for windows without frauds** (`nearest_fallback=True`). If no fraud falls inside a window, it takes up
   to `cap` frauds nearest in time to the window start (half before, half after). The manifest flags this per concept
   (`used_nearest_fallback`).
6. **Remove the time column.** `TransactionDT` is not a feature.
7. **Output.** `__train.parquet` and `__test.parquet` with columns `concept_id, concept_name (Concept_0…Concept_7),
   label, <features>`. Each concept is tested on its own test set only.

So each concept has: a train set (normal only, earliest 60% of the window) and a test set (later 40% of the
normals plus that window's frauds). Expected test fraud rate is about 2,582 / (28,494 + 2,582) ≈ 8%, against 3.5% in
the raw data. The manifest has the real numbers per concept.

## 4. Scenario and evaluation (pyCLAD)

- **Scenario `concept_incremental`** (`pyclad/scenarios/concept_incremental.py`): for each concept in time order,
  the strategy learns that concept's training set. The current model is then evaluated on the test set of **every**
  concept, including future ones. This produces a matrix `M[i][j]` = score on concept *j* after learning concept *i*
  (rows: "after learning", columns: "evaluated on").
- **Strategies** (`lfd/registry.py`):
  - `Naive` re-fits a fresh model on the new concept only (lowest cost).
  - `Cumulative` re-fits on all data seen so far (highest cost; the retention reference).
  - `ReplayEnhanced` fits on the new concept plus a replay buffer of earlier normal rows (adaptive-balanced buffer,
    random selection, `buffer_size` rows), then adds the new concept to the buffer.
  - `ReplayOnly` fits on the buffer only.
  - `MSTE` keeps one model per concept and is told the concept id at test time. It is an oracle, available only in
    `concept_aware`, and must not be compared head-to-head without saying so.
- **Base metrics:** AP (average precision; leads for rare fraud), ROC-AUC, F1.
- **Summary metrics over the matrix** (pyCLAD + `lfd/metrics.py`):

  | name | cells used | question it answers |
  |---|---|---|
  | ContinualAverage (pyCLAD) | diagonal + below-diagonal, `j ≤ i` | overall: just-learned and earlier concepts together |
  | DiagonalAverage (lfd) | `M[i][i]` | how good is the model on the concept it has just learned? ("knowledge accumulation") |
  | PastConceptsAverage (lfd) | `M[i][j]`, `j < i` | how much of the earlier concepts is retained? |
  | BackwardTransfer (pyCLAD) | change on old concepts after each new one | negative = forgetting |
  | ForgettingMeasureStrict (pyCLAD) | best-ever minus current, old concepts | higher = more forgetting |
  | FinalStepAverage (pyCLAD) | last row | performance after the whole stream |

  ContinualAverage does **not** measure only entries below the diagonal. It includes the diagonal. The two lfd
  metrics split it into its two parts. With *N* concepts and no missing cells:
  `ContinualAverage = (N·Diagonal + N(N−1)/2·Past) / (N + N(N−1)/2)`.
  Cells above the diagonal (concepts not yet learned) are not in any of these averages. Since lfd 0.3.0 they are
  also hidden in heatmaps by default.
- **Cost:** `samples_fitted_total` (rows passed to `fit` across the stream), plus fit seconds and wall time
  (`lfd/cost.py`).

## 5. Known caveats of prep_v01 (kept on purpose for faithfulness; fixes go into prep_v02 after approval)

- **BL-0001: statistics fit on all data.** Scaler, encoders, medians and group statistics see future concepts.
  This leaks information from later periods into early concepts. Fix: fit on past data only.
- **BL-0002: test frauds come from the training period.** Frauds are taken from the whole window and the
  *earliest* are kept, so they mostly fall in the first 60% of the window (the training period). The test normals
  come from the last 40%. Test frauds and test normals therefore also differ in time, not only in "fraudness".
  Fix available: `TimeStreamConfig(anomaly_span="test_period")`.
- **BL-0003: fallback may share frauds.** When a window has no frauds, the nearest-in-time fallback can borrow
  frauds that belong to another window, so two concepts may be tested on the same frauds. Fix: disable the fallback
  and merge or report fraud-free windows.
- Equal-count windows have unequal durations (`window="equal_time"` is the alternative).
- The 8 concepts of prep_v01 differ from the "7 concepts" mentioned in the 2026-10-12 meeting. Confirm which
  stream the reported results came from before writing the paper.

## 6. How to cite / paraphrase

> We order the IEEE-CIS transactions by `TransactionDT` and split the normal transactions into eight consecutive
> windows of equal size. Each window is one concept. The earliest 60% of a window's normal transactions form its
> (normal-only) training set; the remaining 40%, together with the frauds that occurred in the window (at most
> ⌊#frauds/8⌋ per concept), form its test set. Features follow the artgor kernel (amount/identity ratios,
> e-mail-domain splits, label encoding, median imputation, standardisation). Models are trained in a
> concept-incremental scenario with pyCLAD (Faber et al., SoftwareX 29, 2025, 101994) and evaluated on every concept
> after each step.

State the caveats of §5 in the threats-to-validity paragraph if prep_v01 numbers are reported.
