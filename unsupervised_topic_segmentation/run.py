"""Command-line entry point for the embedding-based segmentation baseline.

Examples:

    # Run the S-BERT segmenter on every CSV in ../pipeline/to-label/
    python run.py --algorithm sbert --input-dir ../pipeline/to-label

    # Run on the in-person-adult dataset and write labeled output
    python run.py --algorithm sbert \\
        --input-dir ../data/in-person-adult/in-person-adult-human \\
        --output-dir segmented-out/in-person-adult

    # Run all three algorithms and print side-by-side averages
    python run.py --algorithm all --input-dir ../pipeline/to-label
"""

from __future__ import annotations

import argparse
import os
from typing import Dict

import pandas as pd

from core import topic_segmentation
from dataset import MEETING_ID_COL, load_directory, load_files
from eval import evaluate, write_labeled_csvs
from seg_types import (
    TextTilingHyperparameters,
    TopicSegmentationAlgorithm,
    TopicSegmentationConfig,
)


_ALG_CHOICES = {
    "random": TopicSegmentationAlgorithm.RANDOM,
    "even": TopicSegmentationAlgorithm.EVEN,
    "bert": TopicSegmentationAlgorithm.BERT,
    "sbert": TopicSegmentationAlgorithm.SBERT,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input-dir", help="Directory of project CSVs to segment.")
    src.add_argument("--input-files", nargs="+", help="Explicit list of CSVs to segment.")

    p.add_argument(
        "--algorithm",
        choices=list(_ALG_CHOICES.keys()) + ["all"],
        default="sbert",
        help="Segmentation algorithm to run (default: sbert).",
    )
    p.add_argument("--output-dir", default=None,
                   help="If set, write per-file segmented CSVs here.")

    p.add_argument("--window", type=int, default=10,
                   help="TextTiling sentence-comparison window size k (default: 10).")
    p.add_argument("--smoothing-passes", type=int, default=2,
                   help="Number of smoothing passes (default: 2).")
    p.add_argument("--smoothing-window", type=int, default=1,
                   help="Smoothing half-width (default: 1).")
    p.add_argument("--threshold", type=float, default=0.6,
                   help="Topic-change depth threshold × max-depth (default: 0.6).")
    p.add_argument("--cap-segments", action="store_true",
                   help="Cap segment count by expected average length.")

    p.add_argument("--quiet", action="store_true", help="Suppress per-file printing.")
    return p


def _print_pk_summary(name: str, pk_m: Dict):
    avg = pk_m["average"]
    print(f"  {name:<8}  Pk = {avg['pk']:.4f}    WinDiff = {avg['windiff']:.4f}")


def _print_pipeline_summary(pipe_m: Dict):
    o = pipe_m["overall"]
    print("  Line-level metrics (matches pipeline/analysis.py):")
    print(f"    START     Acc={o['start']['accuracy']:.4f}  P={o['start']['precision']:.4f}  "
          f"R={o['start']['recall']:.4f}  F1={o['start']['f1']:.4f}")
    print(f"    END       Acc={o['end']['accuracy']:.4f}  P={o['end']['precision']:.4f}  "
          f"R={o['end']['recall']:.4f}  F1={o['end']['f1']:.4f}")
    print(f"    SEGMENT   Acc={o['segment']['accuracy']:.4f}  P={o['segment']['precision']:.4f}  "
          f"R={o['segment']['recall']:.4f}  F1={o['segment']['f1']:.4f}  IoU={o['segment']['iou']:.4f}")


def run_one(
    df: pd.DataFrame,
    algorithm: TopicSegmentationAlgorithm,
    cfg: TopicSegmentationConfig,
    output_dir: str | None,
    quiet: bool,
) -> Dict[str, Dict]:
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm.name}")
    print(f"{'='*60}")

    boundaries = topic_segmentation(
        algorithm,
        df,
        meeting_id_col_name=MEETING_ID_COL,
        start_col_name="st",
        end_col_name="en",
        caption_col_name="caption",
        topic_segmentation_config=cfg,
    )

    predicted_binary, pk_m, pipe_m = evaluate(df, boundaries)

    if not quiet:
        print("\n  Per-file Pk / WinDiff:")
        for mid, vals in sorted(pk_m["per_file"].items()):
            print(f"    {mid:<24}  Pk={vals['pk']:.4f}  WinDiff={vals['windiff']:.4f}")

    print("\n  AVERAGE:")
    _print_pk_summary("", pk_m)
    _print_pipeline_summary(pipe_m)

    if output_dir:
        suffix = algorithm.name.lower()
        alg_dir = os.path.join(output_dir, suffix)
        written = write_labeled_csvs(df, predicted_binary, alg_dir)
        print(f"\n  Wrote {len(written)} labeled CSV(s) to {alg_dir}")

    return {"boundaries": boundaries, "pk": pk_m, "pipeline": pipe_m}


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.input_dir:
        df = load_directory(args.input_dir)
    else:
        df = load_files(args.input_files)

    print(f"Loaded {df[MEETING_ID_COL].nunique()} transcript(s), "
          f"{len(df)} total utterances.")

    cfg = TopicSegmentationConfig(
        TEXT_TILING=TextTilingHyperparameters(
            SENTENCE_COMPARISON_WINDOW=args.window,
            SMOOTHING_PASSES=args.smoothing_passes,
            SMOOTHING_WINDOW=args.smoothing_window,
            TOPIC_CHANGE_THRESHOLD=args.threshold,
        ),
        MAX_SEGMENTS_CAP=args.cap_segments,
    )

    if args.algorithm == "all":
        algorithms = [
            TopicSegmentationAlgorithm.RANDOM,
            TopicSegmentationAlgorithm.EVEN,
            TopicSegmentationAlgorithm.BERT,
            TopicSegmentationAlgorithm.SBERT,
        ]
    else:
        algorithms = [_ALG_CHOICES[args.algorithm]]

    summary: Dict[str, Dict] = {}
    for alg in algorithms:
        summary[alg.name] = run_one(df, alg, cfg, args.output_dir, args.quiet)

    if len(algorithms) > 1:
        print(f"\n{'='*60}\nCOMPARISON SUMMARY\n{'='*60}")
        print(f"  {'Algorithm':<10} {'Pk':>8} {'WinDiff':>10} {'Seg F1':>10} {'Seg IoU':>10}")
        print(f"  {'-'*10} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
        for name, r in summary.items():
            pk_avg = r["pk"]["average"]
            seg = r["pipeline"]["overall"]["segment"]
            print(f"  {name:<10} {pk_avg['pk']:>8.4f} {pk_avg['windiff']:>10.4f} "
                  f"{seg['f1']:>10.4f} {seg['iou']:>10.4f}")


if __name__ == "__main__":
    main()
