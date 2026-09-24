#!/usr/bin/env python3
"""Store breakpoint alignments for FusOn-DB catalog pairings.

The catalog already holds every head×tail pairing. This job fills the
alignments table for pairings that do not have a result yet, and it can be
stopped and started again.

Reviewed-only runs the priority set: both the head and tail UniProt entries
are reviewed.
"""

import argparse

from fusondb_catalog import DEFAULT_CATALOG_PATH, align_catalog


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align FusOn-DB catalog pairings and store breakpoint results."
    )
    parser.add_argument("--db", default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument(
        "--reviewed-only",
        action="store_true",
        help="Align only pairings whose head and tail entries are both reviewed.",
    )
    parser.add_argument(
        "--limit-fusions",
        type=int,
        default=None,
        help="Stop after this many oncoproteins. Omit to run everything still missing.",
    )
    parser.add_argument("--match", type=int, default=2)
    parser.add_argument("--mismatch", type=int, default=-1)
    parser.add_argument("--gap", type=int, default=-2)
    parser.add_argument("--inter-fragment-gap-penalty", type=int, default=1)
    return parser.parse_args()


def main():
    args = parse_args()
    totals = align_catalog(
        args.db,
        reviewed_only=args.reviewed_only,
        limit_fusions=args.limit_fusions,
        find_kwargs={
            "match": args.match,
            "mismatch": args.mismatch,
            "gap": args.gap,
            "inter_fragment_gap_penalty": args.inter_fragment_gap_penalty,
        },
    )
    print(
        "Done. "
        f"Fusions touched: {totals['fusions']:,}. "
        f"Pairings stored: {totals['pairings']:,} "
        f"({totals['ok']:,} ok, {totals['failed']:,} failed)."
    )


if __name__ == "__main__":
    main()
