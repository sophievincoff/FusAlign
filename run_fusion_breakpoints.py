#!/usr/bin/env python3

import argparse
from pathlib import Path

import pandas as pd

from fusion_breakpoint_pipeline import run_fusion_breakpoint_batch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fusion breakpoint analysis from the command line."
    )

    parser.add_argument(
        "--input-csv",
        required=True,
        help="Input CSV containing fusion, head, tail, and fusion name columns.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for HTML and CSV results.",
    )

    parser.add_argument("--fusion-col", default="fusion")
    parser.add_argument("--head-col", default="head")
    parser.add_argument("--tail-col", default="tail")
    parser.add_argument("--fusion-name-col", default="fusion_name")

    parser.add_argument(
        "--combined-html-name",
        default="fusion_breakpoint_report.html",
        help="Filename for the combined HTML report.",
    )
    parser.add_argument(
        "--combined-csv-name",
        default="fusion_breakpoint_summary.csv",
        help="Filename for the combined CSV summary.",
    )

    parser.add_argument(
        "--write-individual-html",
        action="store_true",
        help="Write one HTML file per fusion protein.",
    )

    parser.add_argument("--head-color", default="crimson")
    parser.add_argument("--tail-color", default="royalblue")
    parser.add_argument("--decimals", type=int, default=2)

    parser.add_argument("--match", type=int, default=2)
    parser.add_argument("--mismatch", type=int, default=-1)
    parser.add_argument("--gap", type=int, default=-2)
    parser.add_argument("--inter-fragment-gap-penalty", type=int, default=1)

    return parser.parse_args()


def main():
    args = parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV does not exist: {input_csv}")

    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    find_kwargs = {
        "match": args.match,
        "mismatch": args.mismatch,
        "gap": args.gap,
        "inter_fragment_gap_penalty": args.inter_fragment_gap_penalty,
    }

    results_df, html_path, csv_path = run_fusion_breakpoint_batch(
        df,
        fusion_col=args.fusion_col,
        head_col=args.head_col,
        tail_col=args.tail_col,
        fusion_name_col=args.fusion_name_col,
        out_dir=out_dir,
        combined_html_name=args.combined_html_name,
        combined_csv_name=args.combined_csv_name,
        write_per_fusion_html=args.write_individual_html,
        display_combined=False,
        head_color=args.head_color,
        tail_color=args.tail_color,
        find_kwargs=find_kwargs,
        decimals=args.decimals,
    )

    n_total = len(results_df)
    n_ok = int((results_df["status"] == "ok").sum()) if "status" in results_df else 0
    n_failed = n_total - n_ok

    print("Done.")
    print(f"Input CSV: {input_csv.resolve()}")
    print(f"Output directory: {out_dir.resolve()}")
    print(f"Combined HTML: {Path(html_path).resolve()}")
    print(f"Combined CSV: {Path(csv_path).resolve()}")
    print(f"Total fusions: {n_total}")
    print(f"Successful: {n_ok}")
    print(f"Failed: {n_failed}")

    if args.write_individual_html:
        print(f"Individual HTML directory: {(out_dir / 'individual_fusions').resolve()}")


if __name__ == "__main__":
    main()