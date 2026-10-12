"""Approval gate. Agents PROPOSE; only the researcher APPROVES.

A proposal is a markdown file with a small header:

    ---
    id: 20261012-010203__data__ieee-fraud__monthly-concepts
    kind: data | improvement
    dataset: ieee-fraud
    status: pending | approved | rejected
    approved_at:
    ---
    (body: what, why, expected effect, exact steps)

CLI:
    python -m lfd.proposals new --kind data --dataset ieee-fraud --title "monthly concepts" --body-file tmp.md
    python -m lfd.proposals list [--dataset X] [--status pending]
    python -m lfd.proposals approve <path>     # .claude/settings.json makes this ASK the human every time
    python -m lfd.proposals reject  <path> --reason "..."
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path

from lfd import naming

HEADER_RE = re.compile(r"^---\n(.*?)\n---\n", re.S)


def _read(path: Path) -> tuple[dict, str]:
    text = Path(path).read_text(encoding="utf-8")
    m = HEADER_RE.match(text)
    if not m:
        raise ValueError(f"{path} has no proposal header")
    header = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            header[k.strip()] = v.strip()
    return header, text[m.end():]


def _write(path: Path, header: dict, body: str) -> None:
    head = "\n".join(f"{k}: {v}" for k, v in header.items())
    Path(path).write_text(f"---\n{head}\n---\n{body}", encoding="utf-8")


def new(kind: str, dataset: str, title: str, body: str) -> Path:
    pid = f"{naming.timestamp()}__{kind}__{naming.slug(dataset, 40)}__{naming.slug(title, 50)}"
    path = naming.PROPOSALS / naming.slug(dataset, 40) / f"{pid}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    _write(path, {"id": pid, "kind": kind, "dataset": dataset, "status": "pending",
                  "approved_at": "", "title": title}, body if body.endswith("\n") else body + "\n")
    return path


def require_approved(path: str | Path, kind: str | None = None) -> dict:
    """Call at the top of anything that changes data or launches a non-baseline experiment."""
    header, _ = _read(Path(path))
    if header.get("status") != "approved":
        sys.exit(f"BLOCKED: proposal {path} has status '{header.get('status')}'. "
                 f"The researcher must approve it first (python -m lfd.proposals approve {path}).")
    if kind and header.get("kind") != kind:
        sys.exit(f"BLOCKED: proposal {path} is kind '{header.get('kind')}', expected '{kind}'.")
    return header


def _set_status(path: Path, status: str, reason: str = "") -> None:
    header, body = _read(path)
    header["status"] = status
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    if status == "approved":
        header["approved_at"] = stamp
    else:
        body += f"\n> {status} {stamp}: {reason}\n"
    _write(path, header, body)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--kind", choices=["data", "improvement"], required=True)
    n.add_argument("--dataset", required=True)
    n.add_argument("--title", required=True)
    n.add_argument("--body-file", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--dataset")
    ls.add_argument("--status")
    a = sub.add_parser("approve")
    a.add_argument("path")
    r = sub.add_parser("reject")
    r.add_argument("path")
    r.add_argument("--reason", default="")
    args = ap.parse_args()

    if args.cmd == "new":
        print(new(args.kind, args.dataset, args.title, Path(args.body_file).read_text(encoding="utf-8")))
    elif args.cmd == "list":
        for p in sorted(naming.PROPOSALS.rglob("*.md")):
            h, _ = _read(p)
            if (args.dataset and h.get("dataset") != args.dataset) or (args.status and h.get("status") != args.status):
                continue
            print(f"[{h.get('status'):8}] {h.get('kind'):11} {p.relative_to(naming.ROOT)}")
    elif args.cmd == "approve":
        _set_status(Path(args.path), "approved")
        print(f"approved: {args.path}")
    elif args.cmd == "reject":
        _set_status(Path(args.path), "rejected", args.reason)
        print(f"rejected: {args.path}")


if __name__ == "__main__":
    main()
