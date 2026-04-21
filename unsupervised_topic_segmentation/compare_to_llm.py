"""Side-by-side comparison of this embedding baseline vs the LLM pipelines.

For each dataset (siblings, in-person-adult), this script:

1. Aggregates per-row stats from the already-written compare CSVs in
   ``../data/<dataset>/<dataset>-compare-hum-*/``, giving us the
   Start/End/Segment metrics for each LLM method (gpt-5.2, gpt-oss-120b).
2. Runs the S-BERT embedding baseline on the human-labeled CSVs of the
   same dataset and computes the same Start/End/Segment metrics plus
   Pk / WinDiff.
3. Also computes Pk / WinDiff for the LLM methods using their predicted
   start=TRUE rows as boundaries, so all methods can be compared on the
   segmentation-specific metrics too.
4. Prints a combined per-dataset table.

Nothing outside ``unsupervised_topic_segmentation/`` is modified.
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Tuple

from nltk.metrics.segmentation import pk, windowdiff

from core import topic_segmentation
from dataset import (
    MEETING_ID_COL, START_COL, END_COL, CAPTION_COL,
    START_LBL_COL, END_LBL_COL, load_directory, human_boundary_labels,
)
from eval import (
    boundaries_to_binary, binary_to_nltk_string,
    compute_pipeline_metrics, compute_pk_windiff,
)
from seg_types import (
    TopicSegmentationAlgorithm, TopicSegmentationConfig, TextTilingHyperparameters,
)


DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


# Each entry: (dataset label, human CSV dir, [(method label, compare-csv dir or raw-LLM dir, kind)])
# kind = "compare" if the dir holds already-merged *_compare.csv files,
#        "raw"     if it holds raw LLM-labeled *_labeled.csv files and we need to pair them with human files.
DATASETS = [
    (
        "siblings",
        os.path.join(DATA_ROOT, "siblings", "siblings-human"),
        [
            ("gpt-5.2",     os.path.join(DATA_ROOT, "siblings", "siblings-compare-hum-5.2"),  "compare"),
            ("gpt-oss-120b", os.path.join(DATA_ROOT, "siblings", "siblings-compare-hum-oss"),  "compare"),
        ],
    ),
    (
        "in-person-adult",
        os.path.join(DATA_ROOT, "in-person-adult", "in-person-adult-human"),
        [
            ("gpt-5.2",     os.path.join(DATA_ROOT, "in-person-adult", "in-person-adult-gpt-5.2"),    "raw"),
            ("gpt-oss-120b", os.path.join(DATA_ROOT, "in-person-adult", "in-person-adult-compare-hum-oss"), "compare"),
        ],
    ),
]


# --- Metric aggregation helpers -----------------------------------------

def _metrics(tp, tn, fp, fn):
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def _aggregate_compare_csvs(compare_dir: str) -> Tuple[Dict, List[Tuple[str, List[int], List[int]]]]:
    """Walk every ``*_compare.csv`` in ``compare_dir`` and aggregate stats.

    Returns:
        metrics dict (start/end/segment with accuracy/precision/recall/f1 + segment IoU)
        per-file (meeting_id, human_boundaries_binary, llm_boundaries_binary) list
            — used to feed Pk / WinDiff computation.
    """
    s = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    e = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    g = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    inter = union = 0
    files_info: List[Tuple[str, List[int], List[int]]] = []

    for path in sorted(glob.glob(os.path.join(compare_dir, "*.csv"))):
        mid = os.path.basename(path).rsplit("_compare.csv", 1)[0]
        with open(path, "r", encoding="utf-8") as f:
            rdr = csv.reader(f)
            rows = list(rdr)[1:]
        hb: List[int] = []
        lb: List[int] = []
        for r in rows:
            if len(r) < 10:
                continue
            sh, sl, eh, el, hseg, lseg = (
                r[1].upper() == "TRUE", r[2].upper() == "TRUE",
                r[3].upper() == "TRUE", r[4].upper() == "TRUE",
                r[6].upper() == "TRUE", r[7].upper() == "TRUE",
            )
            # START
            if sh and sl: s["tp"] += 1
            elif sl and not sh: s["fp"] += 1
            elif sh and not sl: s["fn"] += 1
            else: s["tn"] += 1
            # END
            if eh and el: e["tp"] += 1
            elif el and not eh: e["fp"] += 1
            elif eh and not el: e["fn"] += 1
            else: e["tn"] += 1
            # SEGMENT
            if hseg and lseg: g["tp"] += 1
            elif lseg and not hseg: g["fp"] += 1
            elif hseg and not lseg: g["fn"] += 1
            else: g["tn"] += 1
            if hseg and lseg: inter += 1
            if hseg or lseg: union += 1

            hb.append(1 if sh else 0)
            lb.append(1 if sl else 0)
        # Convention: drop the first-row "boundary" for Pk/WinDiff
        if hb: hb[0] = 0
        if lb: lb[0] = 0
        files_info.append((mid, hb, lb))

    metrics = {
        "start": _metrics(**s),
        "end": _metrics(**e),
        "segment": {**_metrics(**g), "iou": inter / union if union else 0.0},
    }
    return metrics, files_info


def _aggregate_raw_llm_vs_human(human_dir: str, llm_dir: str) -> Tuple[Dict, List[Tuple[str, List[int], List[int]]]]:
    """Same as ``_aggregate_compare_csvs`` but computes from raw human + LLM CSVs.

    Used when the corresponding compare folder is empty.
    """
    s = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    e = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    g = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    inter = union = 0
    files_info: List[Tuple[str, List[int], List[int]]] = []

    for human_path in sorted(glob.glob(os.path.join(human_dir, "*.csv"))):
        mid = os.path.splitext(os.path.basename(human_path))[0]
        llm_path = os.path.join(llm_dir, f"{mid}_labeled.csv")
        if not os.path.exists(llm_path):
            continue
        with open(human_path, "r", encoding="utf-8") as f:
            hrows = list(csv.reader(f))[1:]
        with open(llm_path, "r", encoding="utf-8") as f:
            lrows = list(csv.reader(f))[1:]
        n = min(len(hrows), len(lrows))

        toggle_h = toggle_l = False
        hb: List[int] = []
        lb: List[int] = []
        for i in range(n):
            hr, lr = hrows[i], lrows[i]
            if len(hr) < 3 or len(lr) < 3:
                continue
            sh = hr[1].upper() == "TRUE"
            sl = lr[1].upper() == "TRUE"
            eh = hr[2].upper() == "TRUE"
            el = lr[2].upper() == "TRUE"

            if sh: toggle_h = True
            hseg = toggle_h
            if eh: toggle_h = False
            if sl: toggle_l = True
            lseg = toggle_l
            if el: toggle_l = False

            if sh and sl: s["tp"] += 1
            elif sl and not sh: s["fp"] += 1
            elif sh and not sl: s["fn"] += 1
            else: s["tn"] += 1

            if eh and el: e["tp"] += 1
            elif el and not eh: e["fp"] += 1
            elif eh and not el: e["fn"] += 1
            else: e["tn"] += 1

            if hseg and lseg: g["tp"] += 1
            elif lseg and not hseg: g["fp"] += 1
            elif hseg and not lseg: g["fn"] += 1
            else: g["tn"] += 1
            if hseg and lseg: inter += 1
            if hseg or lseg: union += 1

            hb.append(1 if sh else 0)
            lb.append(1 if sl else 0)

        if hb: hb[0] = 0
        if lb: lb[0] = 0
        files_info.append((mid, hb, lb))

    metrics = {
        "start": _metrics(**s),
        "end": _metrics(**e),
        "segment": {**_metrics(**g), "iou": inter / union if union else 0.0},
    }
    return metrics, files_info


def _pk_windiff_from_file_infos(files_info: List[Tuple[str, List[int], List[int]]]) -> Dict[str, float]:
    pks: List[float] = []
    wds: List[float] = []
    for mid, hb, lb in files_info:
        if not hb or hb.count(1) == 0:
            continue
        ref_s = binary_to_nltk_string(hb)
        pred_s = binary_to_nltk_string(lb[: len(hb)])
        k = max(2, int(round(len(ref_s) / (ref_s.count("1") * 2.0))))
        if k >= len(ref_s):
            k = len(ref_s) - 1
        pks.append(pk(ref_s, pred_s))
        wds.append(windowdiff(ref_s, pred_s, k))
    return {
        "pk": sum(pks) / len(pks) if pks else float("nan"),
        "windiff": sum(wds) / len(wds) if wds else float("nan"),
    }


# --- Embedding-baseline pipeline ---------------------------------------

def _run_sbert(human_dir: str, window: int = 10, threshold: float = 0.6) -> Tuple[Dict, Dict]:
    df = load_directory(human_dir)
    cfg = TopicSegmentationConfig(
        TEXT_TILING=TextTilingHyperparameters(
            SENTENCE_COMPARISON_WINDOW=window,
            SMOOTHING_PASSES=2,
            SMOOTHING_WINDOW=1,
            TOPIC_CHANGE_THRESHOLD=threshold,
        ),
    )
    boundaries = topic_segmentation(
        TopicSegmentationAlgorithm.SBERT, df,
        MEETING_ID_COL, START_COL, END_COL, CAPTION_COL, cfg,
    )

    predicted_binary: Dict[str, List[int]] = {}
    for meeting_id in sorted(set(df[MEETING_ID_COL])):
        n = int((df[MEETING_ID_COL] == meeting_id).sum())
        predicted_binary[meeting_id] = boundaries_to_binary(
            boundaries.get(meeting_id, []), n
        )

    ref = human_boundary_labels(df)
    pk_m = compute_pk_windiff(predicted_binary, ref)
    pipe_m = compute_pipeline_metrics(df, predicted_binary)
    return pk_m, pipe_m


# --- Output -------------------------------------------------------------

def _print_table(title: str, rows: List[Tuple]):
    print(f"\n{'='*90}\n{title}\n{'='*90}")
    header = ("Method", "Pk", "WinDiff", "Start F1", "End F1", "Seg F1", "Seg IoU")
    print("  {:<14} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(*header))
    print("  " + "-" * 82)
    for r in rows:
        name, pk_, wd, sf1, ef1, gf1, iou = r
        def f(x):
            return "   n/a " if x is None else f"{x:>8.4f}"
        print(f"  {name:<14} {f(pk_)} {f(wd):>10} {f(sf1):>10} {f(ef1):>10} {f(gf1):>10} {f(iou):>10}")


def main():
    for ds_label, human_dir, llm_sources in DATASETS:
        print(f"\n{'#'*90}\n#  Dataset: {ds_label}\n{'#'*90}")

        rows: List[Tuple] = []

        # LLM methods
        for method_label, src_dir, kind in llm_sources:
            if not os.path.isdir(src_dir) or not glob.glob(os.path.join(src_dir, "*.csv")):
                print(f"  [skip] {method_label}: no files in {src_dir}")
                continue
            if kind == "compare":
                m, files_info = _aggregate_compare_csvs(src_dir)
            else:
                m, files_info = _aggregate_raw_llm_vs_human(human_dir, src_dir)
            pk_m = _pk_windiff_from_file_infos(files_info)
            rows.append((
                method_label,
                pk_m["pk"], pk_m["windiff"],
                m["start"]["f1"], m["end"]["f1"],
                m["segment"]["f1"], m["segment"]["iou"],
            ))

        # Embedding baseline
        print(f"\n  Running S-BERT embedding baseline on {human_dir} ...")
        pk_m, pipe_m = _run_sbert(human_dir)
        o = pipe_m["overall"]
        rows.append((
            "sbert (ours)",
            pk_m["average"]["pk"], pk_m["average"]["windiff"],
            o["start"]["f1"], o["end"]["f1"],
            o["segment"]["f1"], o["segment"]["iou"],
        ))

        _print_table(f"{ds_label}: human-vs-method agreement", rows)


if __name__ == "__main__":
    main()
