"""Scan an exported bundle for anything that identifies a family's rules.

    python tools/bundle_leak_scan.py BUNDLE_DIR --family momentum

The deny-list is built from the family itself: parameter names, reason codes,
symbols, module and function names. Strategy ids and titles in the publication policy
are allowed. Run it before every showcase ingest; exit 1 on any hit.
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def deny_terms(family: str) -> set[str]:
    terms: set[str] = set()
    for spec_path in (ROOT / "families" / family).rglob("strategy.yaml"):
        spec = yaml.safe_load(spec_path.read_text())
        terms |= {str(k) for k in (spec.get("params") or {})}
        terms |= set(spec.get("reason_codes") or {})
        terms |= {str(s) for s in (spec.get("data") or {}).get("symbols", [])}
    for code in (ROOT / "families" / family).rglob("*.py"):
        tree = ast.parse(code.read_text())
        terms |= {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        terms.add(code.parent.name)
    allowed = set()
    publication = yaml.safe_load((ROOT / "families" / family / "publication.yaml").read_text())
    for item in publication.get("strategies", []):
        allowed |= {item["id"], item["id"].replace("-", "_")}
    return {t for t in terms if len(t) >= 3 and t not in allowed and t not in {"decide", "__init__", "src"}}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--family", required=True)
    args = parser.parse_args()
    terms = deny_terms(args.family)
    hits = []
    for path in sorted(args.bundle.rglob("*")):
        if path.is_file() and path.suffix in {".json", ".md", ".svg"} and path.name != "bundle.json":
            text = path.read_text(errors="replace")
            hits += [
                f"{path.relative_to(args.bundle)}: {t}"
                for t in sorted(terms)
                if re.search(rf"\b{re.escape(t)}\b", text)
            ]
    for hit in hits:
        print(f"BUNDLE_LEAK {hit}")
    print(f"{len(terms)} deny terms; {len(hits)} hit(s)")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
