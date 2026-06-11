#!/usr/bin/env python3
"""List an arXiv paper's section map, or fetch a single section, via DeepXiv.

This is the context-budget tool for arXiv full text. The intended flow:

  1. `fetch_section.py --id <arxiv_id> --list` — print the section map
     (one line per section: `idx | name | token_count | tldr`). One cheap
     `head()` call; no section text is fetched.
  2. Pick the section you need by its TLDR / token count.
  3. `fetch_section.py --id <arxiv_id> --section <name>` — fetch just that
     section's text and quote verbatim. Never load the whole paper into
     context unless you truly need it (use `fetch_paper.py` for that).

Full section text is saved under `$PAPER_NAV_PAPERS_DIR` (as
`<id>__<section>.md`); stdout is truncated to `--max-output` chars.
"""

import argparse
import json
import os
import sys

import deepxiv_client
from fetch_paper import _resolve_papers_dir, _safe_filename
from utils import _strip_arxiv_version


def main():
    parser = argparse.ArgumentParser(
        description="List sections or fetch one section of an arXiv paper via DeepXiv"
    )
    parser.add_argument(
        "--id", required=True, help="arXiv ID (e.g. 2409.05591), version optional"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--list",
        action="store_true",
        help="Print the section map: `idx | name | token_count | tldr`",
    )
    mode.add_argument(
        "--section",
        help="Fetch this section's text (case-insensitive, partial match)",
    )
    parser.add_argument(
        "--max-output",
        type=int,
        default=2000,
        help="Max stdout chars for --section (default 2000). Full text saved to file.",
    )
    parser.add_argument(
        "--papers-dir",
        default=None,
        help="Directory to save section text. Defaults to $PAPER_NAV_PAPERS_DIR. "
        "If unset, --section prints to stdout only (no file write).",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    arxiv_id = _strip_arxiv_version(args.id)

    if args.list:
        head = deepxiv_client.head(arxiv_id)
        if not head:
            print(f"Error: no section map for arXiv:{arxiv_id}", file=sys.stderr)
            sys.exit(1)
        if args.json:
            print(json.dumps(head, indent=2, default=str))
            return
        title = head.get("title", "")
        if title:
            print(f"# {title}\n")
        print(f"arXiv:`{arxiv_id}` | total tokens: {head.get('token_count', '?')}\n")
        print("idx | name | token_count | tldr")
        print(deepxiv_client.format_section_map(head))
        return

    # --section mode
    text = deepxiv_client.section(arxiv_id, args.section)
    if not text:
        print(
            f"Error: section '{args.section}' not found / empty for arXiv:{arxiv_id}. "
            "Run with --list to see available sections.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.json:
        print(json.dumps({"arxiv_id": arxiv_id, "section": args.section, "content": text}, default=str))
        return

    papers_dir = _resolve_papers_dir(args.papers_dir)
    filepath = None
    if papers_dir:
        os.makedirs(papers_dir, exist_ok=True)
        filename = _safe_filename(f"{arxiv_id}__{args.section}") + ".md"
        filepath = os.path.join(papers_dir, filename)
        with open(filepath, "w") as f:
            f.write(f"# arXiv:{arxiv_id} — section: {args.section}\n\n")
            f.write(text)
        print(f"Section text ({len(text)} chars) saved to: {filepath}", file=sys.stderr)

    if len(text) > args.max_output:
        print(text[: args.max_output])
        print(f"\n---\n*[Truncated at {args.max_output} chars.", end=" ")
        if filepath:
            print(f"Full section saved to `{filepath}`.]*")
        else:
            print("Set $PAPER_NAV_PAPERS_DIR or --papers-dir to save the full section.]*")
    else:
        print(text)


if __name__ == "__main__":
    main()
