"""One-screen overview of the research state:  python -m lfd.status"""
from __future__ import annotations

import json
from collections import Counter

from lfd import naming
from lfd.proposals import _read


def main() -> None:
    print("RAW DATASETS (data/raw)")
    for d in sorted(p for p in naming.DATA_RAW.iterdir() if p.is_dir()) if naming.DATA_RAW.exists() else []:
        vers = sorted(p.name for p in (naming.DATA_PREP / d.name).glob(f"{d.name}__prep-v*")) \
            if (naming.DATA_PREP / d.name).exists() else []
        print(f"  {d.name:30} preprocessed versions: {len(vers)}" + (f"  latest: {vers[-1]}" if vers else ""))

    print("\nRUNS (results/<dataset>/runs_index.jsonl)")
    for idx in sorted(naming.RESULTS.glob("*/runs_index.jsonl")) if naming.RESULTS.exists() else []:
        rows = [json.loads(line) for line in idx.read_text().splitlines() if line.strip()]
        st = Counter(r["status"] for r in rows)
        sweeps = sorted({r["sweep_id"] for r in rows})
        print(f"  {idx.parent.name:30} runs: {len(rows)} {dict(st)}  sweeps: {len(sweeps)}"
              + (f"  last: {sweeps[-1]}" if sweeps else ""))
        evals = sorted((idx.parent / "evaluations").glob("*")) if (idx.parent / "evaluations").exists() else []
        if evals:
            print(f"  {'':30} last evaluation: {evals[-1].name}")

    print("\nBACKLOG")
    bdir = naming.RESEARCH / "backlog" / "items"
    items = sorted(bdir.glob("BL-*.md")) if bdir.exists() else []
    if items:
        c = Counter(_read(p)[0].get("status") for p in items)
        print("  " + "  ".join(f"{k}: {v}" for k, v in c.items()) + "   (see research/BACKLOG.md)")
    else:
        print("  (empty)")

    print("\nPROPOSALS")
    any_p = False
    for p in sorted(naming.PROPOSALS.rglob("*.md")) if naming.PROPOSALS.exists() else []:
        h, _ = _read(p)
        any_p = True
        print(f"  [{h.get('status'):8}] {p.relative_to(naming.ROOT)}")
    if not any_p:
        print("  (none)")


if __name__ == "__main__":
    main()
