#!/usr/bin/env python3
"""Verbatim evidence ledger for paper claims.

A claim is a statement about a paper backed by a *verbatim snippet proven to
occur in the paper's text*. Use it to record quotes you can cite with
certainty (asta-bench SQA citation metrics, synthetic-question verification,
LitQA2-style needle digging) — never a paraphrase or a hallucinated quote.

Lowest-friction usage — just record a quote. The section is fetched + cached
automatically if you haven't pulled it yet, so there's no fetch-first rule:

    export PAPER_NAV_CLAIMS_LEDGER=/tmp/claims.jsonl
    export PAPER_NAV_PAPERS_DIR=/tmp/papers
    claims.py add 2409.05591 "global memory of long documents" --section method

The snippet is validated (whitespace-normalized substring) against the cached
section text; the claim is written only if it checks out. Read it back with:

    claims.py export                 # human-readable, verified claims
    claims.py export --json          # claim objects for downstream
    claims.py export --recheck       # re-validate vs current cache, flag drift

Every line in the ledger was verified at add time (add rejects unverifiable
snippets and writes nothing), so the stored snippet *is* the durable evidence.
`export --recheck` re-derives validity against the current cache on demand —
status is never stored.

For non-arXiv / arbitrary cached papers (a Jina/Unpaywall DOI paper, or a
`fetch_paper.py` file named by S2 id), point validation at the file directly
with `--source-file PATH`.

Commands:
    add     Validate a snippet and append a claim
    export  Print / filter the ledger (optionally re-validate)

Importable: `from claims import add_claim, load_ledger, validate_snippet`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone

import deepxiv_client
from fetch_paper import _resolve_papers_dir, _safe_filename
from utils import _strip_arxiv_version

LEDGER_ENV = "PAPER_NAV_CLAIMS_LEDGER"


@dataclass
class Claim:
    id: str
    claim: str
    paper_id: str
    section: str
    snippet: str
    timestamp: str
    source_file: str = ""  # only set for --source-file claims


def _normalize(text: str) -> str:
    """Collapse all whitespace runs (incl. newlines) to single spaces."""
    return " ".join(text.split())


def validate_snippet(snippet: str, text: str) -> bool:
    """True if ``snippet`` occurs in ``text`` ignoring whitespace differences.

    Case-sensitive (the words must be verbatim and in order); tolerant of the
    hard line-wraps in DeepXiv / Jina markdown.
    """
    return _normalize(snippet) in _normalize(text)


def claim_id(paper_id: str, section: str, snippet: str) -> str:
    """Deterministic short id for a claim — dedupes identical re-adds."""
    h = hashlib.sha1(f"{paper_id}|{section}|{snippet}".encode())
    return h.hexdigest()[:12]


def _cache_path(papers_dir: str, paper_id: str, section: str) -> str:
    """The cache file for a (paper, section) — same path ``fetch_section`` writes."""
    stem = f"{paper_id}__{section}" if section else paper_id
    return os.path.join(papers_dir, _safe_filename(stem) + ".md")


def _read_source_text(
    paper_id: str,
    section: str,
    *,
    papers_dir: str | None,
    source_file: str | None,
    allow_fetch: bool,
) -> str:
    """Resolve the text a snippet is validated against.

    Order: ``--source-file`` → cached file → DeepXiv fetch (cached as a side
    effect). Raises ``FileNotFoundError`` (no resolvable source) or
    ``LookupError`` (DeepXiv has no text) on miss.
    """
    if source_file:
        try:
            with open(source_file) as f:
                return f.read()
        except OSError as e:
            raise FileNotFoundError(f"--source-file not readable: {source_file} ({e})")

    if not papers_dir:
        raise FileNotFoundError(
            "no papers dir: set PAPER_NAV_PAPERS_DIR, pass --papers-dir, "
            "or use --source-file"
        )

    path = _cache_path(papers_dir, paper_id, section)
    if os.path.exists(path):
        with open(path) as f:
            return f.read()

    if not allow_fetch:
        raise FileNotFoundError(
            f"not cached: {path}. Run fetch_section first, or drop --no-fetch."
        )

    text = (
        deepxiv_client.section(paper_id, section)
        if section
        else deepxiv_client.raw(paper_id)
    )
    if not text:
        where = f" section '{section}'" if section else ""
        raise LookupError(f"DeepXiv returned no text for arXiv:{paper_id}{where}")

    # Cache it under the same name fetch_section uses, so the tools share a cache.
    os.makedirs(papers_dir, exist_ok=True)
    header = f"# arXiv:{paper_id}" + (f" — section: {section}" if section else "")
    with open(path, "w") as f:
        f.write(header + "\n\n")
        f.write(text)
    return text


def load_ledger(path: str | None) -> list[Claim]:
    """Read all claims from a ledger (empty list if it doesn't exist)."""
    if not path or not os.path.exists(path):
        return []
    claims: list[Claim] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            claims.append(
                Claim(
                    id=d.get("id", ""),
                    claim=d.get("claim", ""),
                    paper_id=d.get("paper_id", ""),
                    section=d.get("section", ""),
                    snippet=d.get("snippet", ""),
                    timestamp=d.get("timestamp", ""),
                    source_file=d.get("source_file", ""),
                )
            )
    return claims


def _append(path: str, rec: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def add_claim(
    paper_id: str,
    snippet: str,
    *,
    ledger_path: str,
    claim: str | None = None,
    section: str = "",
    source_file: str | None = None,
    papers_dir: str | None = None,
    allow_fetch: bool = True,
) -> dict:
    """Validate ``snippet`` against the paper's text and append a claim.

    Returns the claim dict. Idempotent — re-adding an identical
    ``(paper_id, section, snippet)`` does not append a second line. Raises
    ``ValueError`` if the snippet does not occur in the source text, or
    ``FileNotFoundError`` / ``LookupError`` if no source text could be resolved.
    """
    paper_id = _strip_arxiv_version(paper_id)
    text = _read_source_text(
        paper_id,
        section,
        papers_dir=papers_dir,
        source_file=source_file,
        allow_fetch=allow_fetch,
    )
    if not validate_snippet(snippet, text):
        where = f" section '{section}'" if section else ""
        raise ValueError(
            f"snippet not found in source for arXiv:{paper_id}{where}. "
            "Check the wording / section name (fetch_section --list)."
        )

    cid = claim_id(paper_id, section, snippet)
    rec: dict = {
        "id": cid,
        "claim": claim if claim is not None else snippet,
        "paper_id": paper_id,
        "section": section,
        "snippet": snippet,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if source_file:
        rec["source_file"] = source_file

    if cid not in {c.id for c in load_ledger(ledger_path)}:
        _append(ledger_path, rec)
    return rec


def resolve_status(claim: Claim, papers_dir: str | None) -> str:
    """Re-validate a claim against its current source, offline.

    Returns ``"verified"`` or ``"broken"``. Uses the stored ``source_file`` if
    present, else the recomputed arXiv cache path. Missing/unreadable source or
    a snippet that no longer matches ⇒ ``"broken"``.
    """
    if claim.source_file:
        path = claim.source_file
    elif papers_dir:
        path = _cache_path(papers_dir, claim.paper_id, claim.section)
    else:
        return "broken"
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return "broken"
    return "verified" if validate_snippet(claim.snippet, text) else "broken"


def _resolve_ledger(cli_arg: str | None) -> str | None:
    return cli_arg or os.environ.get(LEDGER_ENV)


def _require_ledger(cli_arg: str | None) -> str:
    ledger = _resolve_ledger(cli_arg)
    if not ledger:
        print(
            f"Error: no ledger path. Set ${LEDGER_ENV} or pass --ledger.",
            file=sys.stderr,
        )
        sys.exit(2)
    return ledger


def _cmd_add(args) -> None:
    ledger = _require_ledger(args.ledger)
    papers_dir = _resolve_papers_dir(args.papers_dir)
    before = {c.id for c in load_ledger(ledger)}
    try:
        rec = add_claim(
            args.paper_id,
            args.snippet,
            ledger_path=ledger,
            claim=args.claim,
            section=args.section or "",
            source_file=args.source_file,
            papers_dir=papers_dir,
            allow_fetch=not args.no_fetch,
        )
    except (ValueError, FileNotFoundError, LookupError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    loc = f"arXiv:{rec['paper_id']}" + (f" §{rec['section']}" if rec["section"] else "")
    if rec["id"] in before:
        print(f"Already present: {rec['id']} ({loc}) — no change")
    else:
        print(f"Added claim {rec['id']}: {loc}")


def _cmd_export(args) -> None:
    ledger = _require_ledger(args.ledger)
    papers_dir = _resolve_papers_dir(args.papers_dir)
    claims = load_ledger(ledger)

    rows = []
    n_verified = n_broken = 0
    for c in claims:
        status = resolve_status(c, papers_dir) if args.recheck else "verified"
        n_verified += status == "verified"
        n_broken += status == "broken"
        rows.append((c, status))
    if args.recheck:
        print(f"{n_verified} verified, {n_broken} broken", file=sys.stderr)

    selected = [(c, s) for c, s in rows if args.status in ("all", s)]

    if args.json:
        out = []
        for c, s in selected:
            d = {
                "id": c.id,
                "claim": c.claim,
                "paper_id": c.paper_id,
                "section": c.section,
                "snippet": c.snippet,
                "timestamp": c.timestamp,
            }
            if c.source_file:
                d["source_file"] = c.source_file
            if args.recheck:
                d["status"] = s
            out.append(d)
        print(json.dumps(out, indent=2))
        return

    if not selected:
        print("(no matching claims)")
        return
    for c, s in selected:
        loc = f"arXiv:{c.paper_id}" + (f" §{c.section}" if c.section else "")
        tag = "   [broken]" if (args.recheck and s == "broken") else ""
        print(f"- {c.claim}")
        print(f'    "{c.snippet}"')
        print(f"    — {loc}{tag}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verbatim evidence ledger for paper claims"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="Validate a snippet and append a claim")
    a.add_argument("paper_id", help="arXiv id (version optional)")
    a.add_argument("snippet", help="Verbatim evidence quote")
    a.add_argument("--claim", help="The assertion being made (defaults to the snippet)")
    a.add_argument("--section", help="Section name you fetched with (optional)")
    a.add_argument(
        "--source-file",
        help="Validate against this file instead of DeepXiv "
        "(non-arXiv / arbitrary cached papers)",
    )
    a.add_argument(
        "--no-fetch",
        action="store_true",
        help="Don't fetch on a cache miss; require the section already cached",
    )
    a.add_argument("--ledger", help=f"claims.jsonl path (or ${LEDGER_ENV})")
    a.add_argument("--papers-dir", help="Cache dir (or $PAPER_NAV_PAPERS_DIR)")
    a.set_defaults(func=_cmd_add)

    e = sub.add_parser("export", help="Print / filter the ledger")
    e.add_argument(
        "--status",
        choices=["verified", "broken", "all"],
        default="verified",
        help="Filter (default verified). 'broken' is meaningful only with --recheck.",
    )
    e.add_argument(
        "--recheck",
        action="store_true",
        help="Re-validate each claim against the current cache (offline)",
    )
    e.add_argument("--json", action="store_true", help="Emit claim objects as JSON")
    e.add_argument("--ledger", help=f"claims.jsonl path (or ${LEDGER_ENV})")
    e.add_argument("--papers-dir", help="Cache dir (or $PAPER_NAV_PAPERS_DIR)")
    e.set_defaults(func=_cmd_export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
