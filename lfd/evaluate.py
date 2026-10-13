"""Aggregate runs -> tables + figures that test the research claim:
   "strategy S avoids forgetting (close to Cumulative) while costing much less (close to Naive)".

    python -m lfd.evaluate --dataset ieee-fraud                       # all ok runs
    python -m lfd.evaluate --dataset ieee-fraud --sweep <sweep_id> [--sweep <id2>]
    python -m lfd.evaluate --dataset ieee-fraud --metric AP

Output (never overwritten): results/<dataset>/evaluations/<ts>__eval__<scope>/
    <eval>__runs.csv                 every run included (one row per run/seed)
    <eval>__comparison.csv           mean / std / n across seeds per configuration
    <eval>__tradeoff.csv             retention-gain vs cost, normalised to Naive (0) and Cumulative (1)
    <eval>__paired_vs_naive.csv      seed-paired differences vs Naive with 95% CI
    <eval>__cellwise_vs_naive.csv    cells better / equal / worse than Naive (diagonal, past, all), mean over seeds
    <eval>__table_main.csv / .md     paper main table: CA, DiagonalAverage, PastConceptsAverage, BWT, FM, cost
    <eval>__comparison.json          same numbers as comparison.csv, for sharing next to the heatmaps
    <eval>__fig_tradeoff_<scen>_<model>.png/.pdf     the headline figure
    <eval>__fig_cost-vs-perf_<scen>_<model>.png/.pdf
    <eval>__fig_heatmaps_<scen>_<model>.png/.pdf    seed-averaged matrices, one panel per strategy
    <eval>__fig_heatmap-diff-vs-naive_<scen>_<model>.png/.pdf
    <eval>__fig_diagonal_<scen>_<model>.png/.pdf    per-concept M[i][i] (just-learned concept), mean ± std
    <eval>__summary.md               auto tables; the evaluator agent writes <eval>__FINDINGS.md next to it

DiagonalAverage and PastConceptsAverage (lfd/metrics.py) are recomputed post-hoc from each run's saved
metric_matrix, so runs made before they existed are covered without retraining. Heatmaps hide cells above the
diagonal (concepts not learned yet) unless --show-upper-diagonal is given; numbers are not affected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import stats  # noqa: E402

from lfd import naming, registry  # noqa: E402
from lfd.metrics import DiagonalAverage, PastConceptsAverage  # noqa: E402

# fixed colour per strategy (colour follows the entity, never its rank); markers give a 2nd encoding
STRAT_STYLE = {
    "Naive": ("#2a78d6", "o"), "Cumulative": ("#eb6834", "s"), "MSTE": ("#1baf7a", "^"),
    "ReplayOnly": ("#eda100", "D"), "ReplayEnhanced": ("#e87ba4", "v"),
}
EXTRA = [("#008300", "P"), ("#6250d6", "X"), ("#e34948", "*")]
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#86b6ef", "#2a78d6", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", "#2a78d6"])  # red=worse, blue=better
INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e5e1"


def style(strategy: str, i: int = 0):
    base = strategy.split("-")[0]
    return STRAT_STYLE.get(base, EXTRA[i % len(EXTRA)])


def _axes(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def _save(fig, d: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(naming.file_in(d, stem, ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_runs(dataset: str, sweeps: list[str] | None) -> pd.DataFrame:
    idx = naming.RESULTS / naming.slug(dataset, 40) / "runs_index.jsonl"
    if not idx.exists():
        raise SystemExit(f"no runs index at {idx}")
    df = pd.DataFrame([json.loads(line) for line in idx.read_text().splitlines() if line.strip()])
    df = df[df["status"] == "ok"]
    if sweeps:
        df = df[df["sweep_id"].isin(sweeps)]
    if df.empty:
        raise SystemExit("no successful runs match")
    df["config"] = df.apply(lambda r: naming.strategy_tag(r["strategy"], r["strategy_params"]), axis=1)
    # runs with different opt-in features are different settings: never pooled, each has its own Naive/Cumulative
    if "features_tag" not in df.columns:
        df["features_tag"] = ""
    df["setting"] = df["features_tag"].fillna("").replace("", "default")
    return df.reset_index(drop=True)


SUMMARY_COLS = ["ContinualAverage", "BackwardTransfer", "ForwardTransfer", "ForgettingMeasureStrict",
                "FinalStepAverage"]  # stored by every run since 0.1.0
POSTHOC = {"DiagonalAverage": DiagonalAverage(), "PastConceptsAverage": PastConceptsAverage()}  # BL-0005/0006
MAIN_TABLE = ["ContinualAverage", "DiagonalAverage", "PastConceptsAverage", "BackwardTransfer",
              "ForgettingMeasureStrict"]
_MATS: dict = {}


def metric_key(metric: str) -> str:
    """results.json block name for a base metric given by its CLI name (ROC-AUC | AP | F1)."""
    cls = registry.BASE_METRICS.get(metric)
    return f"concept_metric_callback_{cls().name() if cls else metric}"


def run_matrix(path: str, key: str):
    """Square matrix M[learned][evaluated] of one run, rows and columns in learning order. Cached."""
    if (path, key) not in _MATS:
        res = json.loads(next(Path(naming.ROOT / path).glob("*__results.json")).read_text())
        b = res[key]
        order = list(b["concepts_order"])
        mat = np.array([[b["metric_matrix"][r].get(c, np.nan) for c in order] for r in order], dtype=float)
        _MATS[(path, key)] = (mat, order)
    return _MATS[(path, key)]


def mask_upper(mat: np.ndarray) -> np.ndarray:
    out = np.array(mat, dtype=float, copy=True)
    out[np.triu_indices(out.shape[0], 1)] = np.nan
    return out


def mean_matrix(paths: list[str], metric_key: str):
    mats, order = [], None
    for p in paths:
        mat, order = run_matrix(p, metric_key)
        mats.append(mat)
    return np.nanmean(np.array(mats, dtype=float), axis=0), order


def add_posthoc_metrics(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Recompute DiagonalAverage / PastConceptsAverage from every run's saved metric_matrix (no retraining)."""
    key = metric_key(metric)
    vals = {name: [] for name in POSTHOC}
    for p in df["path"]:
        try:
            mat, _ = run_matrix(p, key)
            for name, fn in POSTHOC.items():
                vals[name].append(fn.compute(mat.tolist()))
        except (KeyError, StopIteration, ValueError) as e:
            print(f"warning: no square {metric} matrix for {p} ({type(e).__name__}); new metrics = NaN")
            for name in POSTHOC:
                vals[name].append(np.nan)
    for name, v in vals.items():
        df[f"{metric}__{name}"] = v  # identical to what runs since 0.3.0 store themselves
    return df


REGIONS = {"all": lambda i, j: j <= i, "diagonal": lambda i, j: j == i, "past": lambda i, j: j < i}


def cellwise_vs_naive(g: pd.DataFrame, key: str, tol: float) -> list[dict]:
    """Seed-paired cell counts vs Naive on the diagonal + lower triangle; mean/std over seeds."""
    base = g[g["config"] == "Naive"].set_index("seed")["path"]
    rows = []
    for cfg, gg in g.groupby("config"):
        if cfg == "Naive":
            continue
        per_seed = {r: [] for r in REGIONS}
        for seed, path in gg.set_index("seed")["path"].items():
            if seed not in base.index:
                continue
            try:
                a, oa = run_matrix(path, key)
                b, ob = run_matrix(base.loc[seed] if isinstance(base.loc[seed], str) else base.loc[seed].iloc[0],
                                   key)
            except (KeyError, StopIteration):
                continue
            if oa != ob:
                print(f"warning: concept order differs from Naive for {path}; skipped in cell-wise comparison")
                continue
            diff = a - b
            for region, sel in REGIONS.items():
                cells = [diff[i, j] for i in range(len(oa)) for j in range(len(oa))
                         if sel(i, j) and not np.isnan(diff[i, j])]
                cells = np.array(cells)
                per_seed[region].append((int((cells > tol).sum()), int((np.abs(cells) <= tol).sum()),
                                         int((cells < -tol).sum()), len(cells)))
        for region, counts in per_seed.items():
            if not counts:
                continue
            c = np.array(counts, dtype=float)
            row = {"config": cfg, "region": region, "n_seeds": len(c), "n_cells": c[:, 3].mean()}
            for k, name in enumerate(("better", "equal", "worse")):
                row[f"{name}_mean"] = c[:, k].mean()
                row[f"{name}_std"] = c[:, k].std(ddof=1) if len(c) > 1 else np.nan
                row[f"{name}_frac"] = c[:, k].mean() / c[:, 3].mean() if c[:, 3].mean() else np.nan
            rows.append(row)
    return rows


def _pm(mean, std, nd=4) -> str:
    if pd.isna(mean):
        return "n/a"
    return f"{mean:.{nd}f}" + ("" if pd.isna(std) else f" ± {std:.{nd}f}")


def main_table(comp: pd.DataFrame, keys: list[str], m: str, cost: str) -> pd.DataFrame:
    """Paper main table (BL-0008): one row per configuration, numeric mean/std columns."""
    rows = []
    for _, r in comp.iterrows():
        row = {k: r[k] for k in keys}
        for name in MAIN_TABLE:
            row[f"{name}__mean"] = r[f"{m}__{name}__mean"]
            row[f"{name}__std"] = r[f"{m}__{name}__std"]
        same = (comp[keys[:-1]] == r[keys[:-1]]).all(axis=1) & (comp["config"] == "Cumulative")
        ref = comp.loc[same, f"{cost}__mean"]
        row[f"{cost}__mean"] = r[f"{cost}__mean"]
        row["cost_rel_cumulative"] = r[f"{cost}__mean"] / ref.iloc[0] if len(ref) and ref.iloc[0] else np.nan
        row["n_seeds"] = int(r[f"{m}__ContinualAverage__count"])
        rows.append(row)
    return pd.DataFrame(rows)


def main_table_md(t: pd.DataFrame, m: str, cost: str) -> str:
    keys = ["scenario", "model", "setting", "config"]
    out = pd.DataFrame({k: t[k] for k in keys})
    for name in MAIN_TABLE:
        out[name] = [_pm(a, b) for a, b in zip(t[f"{name}__mean"], t[f"{name}__std"])]
    out["cost / Cumulative"] = t["cost_rel_cumulative"].round(3)
    out["n"] = t["n_seeds"]
    return (f"# Main table · base metric {m} · cost {cost}\n\nmean ± std over seeds. "
            "BackwardTransfer: higher is better; ForgettingMeasureStrict: lower is better.\n\n"
            + out.to_markdown(index=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--sweep", action="append")
    ap.add_argument("--metric", default="ROC-AUC", help="base metric: ROC-AUC | AP | F1")
    ap.add_argument("--cost", default="samples_fitted_total",
                    help="cost column: samples_fitted_total | fit_seconds_total | train_time_s | peak_mem_mb")
    ap.add_argument("--show-upper-diagonal", action="store_true",
                    help="show heatmap cells above the diagonal (concepts not learned yet); hidden by default")
    ap.add_argument("--cell-tol", type=float, default=0.005,
                    help="cell-wise vs Naive: |difference| <= tol counts as equal (default 0.005)")
    args = ap.parse_args()

    df = load_runs(args.dataset, args.sweep)
    m = args.metric
    cols = {k: f"{m}__{k}" for k in SUMMARY_COLS}
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise SystemExit(f"metric {m} not in runs ({missing}); was it listed under `metrics` in the config?")
    df = add_posthoc_metrics(df, m)
    cols.update({k: f"{m}__{k}" for k in POSTHOC})
    mkey = metric_key(m)
    hide_upper = not args.show_upper_diagonal
    scope = "+".join(sorted({s.split("__sweep__")[-1] for s in df["sweep_id"].unique()})) if args.sweep else "all-runs"
    cost_short = {"samples_fitted_total": "cost-samples", "fit_seconds_total": "cost-fitsec",
                  "train_time_s": "cost-traintime", "peak_mem_mb": "cost-mem"}.get(args.cost, args.cost)
    d = naming.eval_dir(args.dataset, f"{naming.slug(scope, 50)}__{naming.slug(m)}__{cost_short}")
    df.to_csv(naming.file_in(d, "runs", "csv"), index=False)

    keys = ["scenario", "model", "setting", "config"]
    agg_cols = list(cols.values()) + [args.cost, "train_time_s", "fit_seconds_total", "peak_mem_mb"]
    comp = df.groupby(keys)[agg_cols].agg(["mean", "std", "count"])
    comp.columns = [f"{a}__{b}" for a, b in comp.columns]
    comp = comp.reset_index()
    comp.to_csv(naming.file_in(d, "comparison", "csv"), index=False)

    ca, bwt, fgt, cost = cols["ContinualAverage"], cols["BackwardTransfer"], cols["ForgettingMeasureStrict"], args.cost
    diag, past = cols["DiagonalAverage"], cols["PastConceptsAverage"]
    trade_rows, paired_rows, cell_rows, md = [], [], [], []
    md.append(f"# Evaluation `{d.name}`\n\nDataset **{args.dataset}** · base metric **{m}** · cost **{cost}** · "
              f"{len(df)} runs · sweeps: {', '.join(sorted(df['sweep_id'].unique()))}\n")

    for (scen, model, setting), g in df.groupby(["scenario", "model", "setting"]):
        mean = g.groupby("config")[[ca, bwt, fgt, cost]].mean()
        has_ref = {"Naive", "Cumulative"} <= set(mean.index)
        tag = f"{scen}_{naming.slug(model)}" + ("" if setting == "default" else f"_{setting}")

        # ---- normalised trade-off: 0 = Naive, 1 = Cumulative, on both axes
        if has_ref:
            n, c = mean.loc["Naive"], mean.loc["Cumulative"]
            for cfg, r in mean.iterrows():
                dc = c[cost] - n[cost]
                dp = c[ca] - n[ca]
                trade_rows.append({
                    "scenario": scen, "model": model, "setting": setting, "config": cfg,
                    "cost_rel_cumulative": r[cost] / c[cost] if c[cost] else np.nan,
                    "cost_norm": (r[cost] - n[cost]) / dc if dc else np.nan,
                    "retention_norm": (r[ca] - n[ca]) / dp if dp else np.nan,
                    "forgetting_avoided_vs_naive": n[fgt] - r[fgt],
                    "bwt_gain_vs_naive": r[bwt] - n[bwt],
                    "ca_gap_to_cumulative": c[ca] - r[ca],
                })
            # paired by seed vs Naive
            base = g[g["config"] == "Naive"].set_index("seed")
            for cfg, gg in g.groupby("config"):
                if cfg == "Naive":
                    continue
                s = gg.set_index("seed").join(base[[ca, bwt, fgt, diag, past]], rsuffix="_naive", how="inner")
                for col in (ca, bwt, fgt, diag, past):
                    diff = (s[col] - s[f"{col}_naive"]).dropna()
                    k = len(diff)
                    ci = stats.t.ppf(0.975, k - 1) * diff.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
                    paired_rows.append({"scenario": scen, "model": model, "setting": setting, "config": cfg,
                                        "metric": col,
                                        "n_seeds": k, "mean_diff_vs_naive": diff.mean(),
                                        "ci95_low": diff.mean() - ci, "ci95_high": diff.mean() + ci,
                                        "significant_95": bool(k > 1 and abs(diff.mean()) > ci)})

            # ---- cell-wise better / equal / worse than Naive (knowledge accumulation, BL-0006)
            for r in cellwise_vs_naive(g, mkey, args.cell_tol):
                cell_rows.append({"scenario": scen, "model": model, "setting": setting, **r})

            # ---- headline figure
            t = pd.DataFrame([r for r in trade_rows
                              if r["scenario"] == scen and r["model"] == model and r["setting"] == setting])
            fig, ax = plt.subplots(figsize=(5.2, 4.4))
            _axes(ax)
            ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", zorder=1)
            ax.fill_between([0, 1], [0, 1], [1.2, 1.2], color="#2a78d6", alpha=0.06, zorder=0)
            ax.text(0.03, 1.12, "more retention than its cost share", color=MUTED, fontsize=7)
            for i, r in t.iterrows():
                col, mk = style(r["config"], i)
                ax.scatter(r["cost_norm"], r["retention_norm"], s=70, color=col, marker=mk,
                           edgecolor="white", linewidth=1.5, zorder=3, label=r["config"])
                ax.annotate(r["config"], (r["cost_norm"], r["retention_norm"]), xytext=(6, 4),
                            textcoords="offset points", fontsize=7, color=INK)
            ax.set_xlabel(f"cost, normalised ({cost}): 0 = Naive, 1 = Cumulative", color=INK, fontsize=8)
            ax.set_ylabel(f"retention ({m} ContinualAverage): 0 = Naive, 1 = Cumulative", color=INK, fontsize=8)
            ax.set_title(f"Retention gained per unit of cost\n{args.dataset} · {scen} · {model}" + ("" if setting == "default" else f" · {setting}"),
                         fontsize=9, color=INK, loc="left")
            ax.legend(fontsize=7, frameon=False, loc="lower right")
            _save(fig, d, f"fig_tradeoff_{tag}")

        # ---- raw cost vs performance (one axis each; log x for cost)
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        _axes(ax)
        for i, (cfg, gg) in enumerate(g.groupby("config")):
            col, mk = style(cfg, i)
            ax.scatter(gg[cost], gg[ca], s=22, color=col, marker=mk, alpha=0.45, edgecolor="none")
            ax.scatter(gg[cost].mean(), gg[ca].mean(), s=80, color=col, marker=mk, edgecolor="white",
                       linewidth=1.5, label=cfg, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel(f"{cost} (log)", color=INK, fontsize=8)
        ax.set_ylabel(f"{m} ContinualAverage", color=INK, fontsize=8)
        ax.set_title(f"Cost vs performance (seeds faint, mean solid)\n{args.dataset} · {scen} · {model}" + ("" if setting == "default" else f" · {setting}"),
                     fontsize=9, color=INK, loc="left")
        ax.legend(fontsize=7, frameon=False)
        _save(fig, d, f"fig_cost-vs-perf_{tag}")

        # ---- heatmap grid (seed-averaged), shared 0..1 scale
        key = mkey
        cfgs = sorted(g["config"].unique(), key=lambda c: list(STRAT_STYLE).index(c.split("-")[0])
                      if c.split("-")[0] in STRAT_STYLE else 99)
        mats = {}
        for cfg in cfgs:
            try:
                mats[cfg] = mean_matrix(g[g["config"] == cfg]["path"].tolist(), key)
            except (KeyError, StopIteration):
                pass
        if mats:
            k = len(mats)
            fig, axs = plt.subplots(1, k, figsize=(3.1 * k, 3.0), squeeze=False)
            for ax, (cfg, (mat, order)) in zip(axs[0], mats.items()):
                if hide_upper:
                    mat = mask_upper(mat)
                im = ax.imshow(mat, cmap=SEQ, vmin=0, vmax=1)
                ax.grid(False)
                ax.set_title(cfg, fontsize=8, color=INK)
                ax.set_xticks(range(len(order)), [str(o)[:8] for o in order], rotation=90, fontsize=6)
                ax.set_yticks(range(len(order)), [str(o)[:8] for o in order], fontsize=6)
                if len(order) <= 10:
                    for (i, j), v in np.ndenumerate(mat):
                        if np.isnan(v):
                            continue
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5,
                                color="white" if v > 0.6 else INK)
            axs[0][0].set_ylabel("after learning concept", fontsize=7)
            fig.supxlabel("evaluated on concept", fontsize=7)
            fig.colorbar(im, ax=axs[0].tolist(), shrink=0.8, label=m)
            fig.suptitle(f"{m} matrices, mean over seeds · {scen} · {model}", fontsize=9, x=0.01, ha="left")
            _save(fig, d, f"fig_heatmaps_{tag}")

            if "Naive" in mats:
                others = [c for c in mats if c != "Naive"]
                if others:
                    fig, axs = plt.subplots(1, len(others), figsize=(3.1 * len(others), 3.0), squeeze=False)
                    base_m = mats["Naive"][0]
                    for ax, cfg in zip(axs[0], others):
                        diff = mats[cfg][0] - base_m
                        if hide_upper:
                            diff = mask_upper(diff)
                        lim = max(0.05, float(np.nanmax(np.abs(diff))))
                        im = ax.imshow(diff, cmap=DIV, vmin=-lim, vmax=lim)
                        ax.grid(False)
                        ax.set_title(f"{cfg} − Naive", fontsize=8, color=INK)
                        order = mats[cfg][1]
                        ax.set_xticks(range(len(order)), [str(o)[:8] for o in order], rotation=90, fontsize=6)
                        ax.set_yticks(range(len(order)), [str(o)[:8] for o in order], fontsize=6)
                        fig.colorbar(im, ax=ax, shrink=0.8)
                    fig.suptitle("Where forgetting is avoided: blue = better than Naive (lower triangle = "
                                 "old concepts)", fontsize=8, x=0.01, ha="left")
                    _save(fig, d, f"fig_heatmap-diff-vs-naive_{tag}")

        # ---- per-concept diagonal M[i][i] (just-learned concept), mean ± std over seeds (BL-0006)
        fig, ax = plt.subplots(figsize=(5.6, 3.8))
        _axes(ax)
        drawn, ref_order = 0, None
        for i, cfg in enumerate(cfgs):
            vecs = []
            for p in g[g["config"] == cfg]["path"]:
                try:
                    mat, order = run_matrix(p, mkey)
                except (KeyError, StopIteration):
                    continue
                ref_order = ref_order or order
                if order == ref_order:
                    vecs.append(np.diag(mat))
            if not vecs:
                continue
            v = np.array(vecs, dtype=float)
            mu = np.nanmean(v, axis=0)
            sd = np.nanstd(v, axis=0, ddof=1) if len(v) > 1 else np.zeros_like(mu)
            col, mk = style(cfg, i)
            x = np.arange(len(mu))
            label = f"{cfg} (oracle concept id)" if cfg.startswith("MSTE") else cfg
            ax.plot(x, mu, color=col, marker=mk, lw=1.5, ms=5, label=f"{label} (n={len(v)})")
            ax.fill_between(x, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
            drawn += 1
        if drawn:
            ax.set_xticks(range(len(ref_order)), [str(o)[:10] for o in ref_order], rotation=45, ha="right")
            ax.set_xlabel("concept just learned (stream order)", color=INK, fontsize=8)
            ax.set_ylabel(f"{m} on that concept, M[i][i]", color=INK, fontsize=8)
            ax.set_title(f"Knowledge accumulation: diagonal of the {m} matrix (mean ± std over seeds)\n"
                         f"{args.dataset} · {scen} · {model}" + ("" if setting == "default" else f" · {setting}"),
                         fontsize=9, color=INK, loc="left")
            ax.legend(fontsize=7, frameon=False)
            _save(fig, d, f"fig_diagonal_{tag}")
        else:
            plt.close(fig)

    trade = pd.DataFrame(trade_rows)
    paired = pd.DataFrame(paired_rows)
    cells = pd.DataFrame(cell_rows)
    if not cells.empty:
        cells.to_csv(naming.file_in(d, "cellwise_vs_naive", "csv"), index=False)
    comp.to_json(naming.file_in(d, "comparison", "json"), orient="records", indent=2)
    table = main_table(comp, keys, m, cost)
    table.to_csv(naming.file_in(d, "table_main", "csv"), index=False)
    naming.file_in(d, "table_main", "md").write_text(main_table_md(table, m, cost), encoding="utf-8")
    if not trade.empty:
        trade.to_csv(naming.file_in(d, "tradeoff", "csv"), index=False)
    if not paired.empty:
        paired.to_csv(naming.file_in(d, "paired_vs_naive", "csv"), index=False)

    show = comp[keys + [f"{c}__mean" for c in cols.values()] + [f"{cost}__mean", f"{ca}__count"]].copy()
    show.columns = keys + list(cols.keys()) + [cost, "n_seeds"]
    md.append("## Mean over seeds\n\n" + show.round(4).to_markdown(index=False) + "\n")
    if not trade.empty:
        md.append("## Trade-off (0 = Naive, 1 = Cumulative)\n\nA strategy supports the claim when "
                  "`retention_norm` is high while `cost_norm` is low (above the diagonal in "
                  "`fig_tradeoff_*`).\n\n" + trade.round(4).to_markdown(index=False) + "\n")
    else:
        md.append("> Trade-off table skipped: needs both Naive and Cumulative runs per scenario/model.\n")
    if not paired.empty:
        md.append("## Seed-paired difference vs Naive (95% CI)\n\n" + paired.round(4).to_markdown(index=False) + "\n")
        if paired["n_seeds"].min() < 5:
            md.append("> ⚠ fewer than 5 seeds for some configurations — CIs are wide; treat as indicative.\n")
    if not cells.empty:
        show_c = cells[["scenario", "model", "setting", "config", "region", "n_cells", "better_mean", "equal_mean",
                        "worse_mean", "n_seeds"]]
        md.append(f"## Cell-wise vs Naive (|diff| ≤ {args.cell_tol} = equal), mean over seed pairs\n\n"
                  "`diagonal` = just-learned concept, `past` = strictly lower triangle, `all` = both "
                  "(the cells of ContinualAverage).\n\n" + show_c.round(2).to_markdown(index=False) + "\n")
    md.append("## Metric definitions\n\n- ContinualAverage: mean of diagonal + strictly-lower triangle "
              "(pyCLAD).\n- DiagonalAverage: mean of M[i][i], the concept just learned (lfd, post-hoc).\n"
              "- PastConceptsAverage: mean of M[i][j], j < i, earlier concepts only (lfd, post-hoc).\n")
    naming.file_in(d, "summary", "md").write_text("\n".join(md), encoding="utf-8")
    print(f"evaluation -> {d.relative_to(naming.ROOT)}")


if __name__ == "__main__":
    main()
