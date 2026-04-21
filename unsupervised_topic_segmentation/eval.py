"""Evaluation + CSV writer for the segmentation baseline.

Two metric families are computed:

1. **Pk / WinDiff** (the paper's metrics).  These credit near-miss
   boundaries within a fixed sliding window rather than requiring
   exact line alignment, so they're the standard segmentation
   measures.  Lower is better.

2. **Line-level precision / recall / F1 / IoU**, matching the metrics
   already used by this project's ``pipeline/analysis.py``, so that
   the embedding baseline and the LLM pipelines can be compared on
   the same axes.  Higher is better.

In addition to scores, this module can emit a **4-column CSV per
transcript** that matches the project's labelled-output format
(``in/out/ambiguous, start, end, Transcript``) so the baseline's
predictions can be fed back through the main pipeline's comparison
tooling if desired.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import pandas as pd
from nltk.metrics.segmentation import pk, windowdiff

from dataset import (
    MEETING_ID_COL,
    START_COL,
    END_COL,
    CAPTION_COL,
    SPEAKER_COL,
    IN_OUT_COL,
    START_LBL_COL,
    END_LBL_COL,
    human_boundary_labels,
)


# --- Boundary ↔ binary-vector conversions --------------------------------

def boundaries_to_binary(boundaries: List[int], length: int) -> List[int]:
    """Convert a list of gap indices into an NLTK-style 0/1 string vector.

    A ``1`` at index ``i`` means "utterance ``i`` is the first utterance
    of a new segment".  Index 0 is always 0 by convention.
    """
    out = [0] * length
    for b in boundaries:
        if 0 < b < length:
            out[b] = 1
    return out


def binary_to_nltk_string(binary: List[int]) -> str:
    """NLTK's Pk/WinDiff functions want a plain string of digits."""
    return "".join(str(b) for b in binary)


# --- Pk / WinDiff -------------------------------------------------------

def compute_pk_windiff(
    predicted_binary: Dict[str, List[int]],
    reference_binary: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """Compute per-transcript and average Pk/WinDiff.

    Returns:
        ``{'per_file': {mid: {'pk': ..., 'windiff': ...}}, 'average': {...}}``
    """
    per_file: Dict[str, Dict[str, float]] = {}
    pks: List[float] = []
    wds: List[float] = []

    for mid, ref in reference_binary.items():
        if mid not in predicted_binary:
            continue
        pred = predicted_binary[mid]
        if len(pred) != len(ref):
            pred = pred[: len(ref)] + [0] * max(0, len(ref) - len(pred))

        ref_s = binary_to_nltk_string(ref)
        pred_s = binary_to_nltk_string(pred)

        num_ref_boundaries = ref_s.count("1")
        if num_ref_boundaries == 0 or len(ref_s) < 2:
            per_file[mid] = {"pk": float("nan"), "windiff": float("nan")}
            continue

        _pk = pk(ref_s, pred_s)
        # k = half the average true segment length (standard convention)
        k = max(2, int(round(len(ref_s) / (num_ref_boundaries * 2.0))))
        if k >= len(ref_s):
            k = len(ref_s) - 1
        _wd = windowdiff(ref_s, pred_s, k)

        per_file[mid] = {"pk": _pk, "windiff": _wd}
        pks.append(_pk)
        wds.append(_wd)

    avg = {
        "pk": sum(pks) / len(pks) if pks else float("nan"),
        "windiff": sum(wds) / len(wds) if wds else float("nan"),
    }
    return {"per_file": per_file, "average": avg}


# --- Line-level confusion-matrix metrics (matches pipeline/analysis.py) --

def _segment_membership(binary: List[int]) -> List[int]:
    """Given a boundary-at-start binary vector, return per-row segment IDs.

    Every row belongs to some segment.  Segment 0 starts at row 0 and
    runs until (exclusive) the first 1, then segment 1 starts, and so on.
    """
    sid = 0
    ids: List[int] = []
    for i, b in enumerate(binary):
        if b == 1 and i != 0:
            sid += 1
        ids.append(sid)
    return ids


def _human_story_membership(df_meeting: pd.DataFrame) -> List[bool]:
    """Per-row boolean: is this row inside any human-annotated story segment?

    Matches the toggle logic in ``pipeline/analysis.py``: turn on at
    ``start_lbl == TRUE``, include the ``end_lbl == TRUE`` row, then
    turn off.
    """
    inside = False
    out: List[bool] = []
    for _, row in df_meeting.iterrows():
        if str(row[START_LBL_COL]).upper() == "TRUE":
            inside = True
        out.append(inside)
        if str(row[END_LBL_COL]).upper() == "TRUE":
            inside = False
    return out


def compute_pipeline_metrics(
    df: pd.DataFrame,
    predicted_binary: Dict[str, List[int]],
) -> Dict[str, Dict]:
    """Compute start-match, end-match, and segment-IoU vs human labels.

    This mirrors the metrics in ``pipeline/analysis.py`` so that
    baseline numbers are directly comparable to the LLM pipeline's
    numbers from the main project.

    For the baseline's predictions, we derive "start" and "end" row
    markers from the predicted boundaries:
        - predicted start = first row of each predicted segment
        - predicted end   = last row of each predicted segment

    Then every row in every predicted segment counts as an in-segment
    row (for IoU), which is a known limitation of the pure-embedding
    approach — it doesn't distinguish story vs non-story, only
    topic-A vs topic-B.  Expect high recall / low precision on the
    segment-IoU metric compared to LLM approaches.
    """
    start_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    end_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    seg_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    inter = 0
    union = 0

    per_file: Dict[str, Dict] = {}

    for meeting_id in sorted(set(df[MEETING_ID_COL])):
        sub = df[df[MEETING_ID_COL] == meeting_id].sort_values(START_COL).reset_index(drop=True)
        n = len(sub)
        pred_bin = predicted_binary.get(meeting_id, [0] * n)
        if len(pred_bin) != n:
            pred_bin = pred_bin[:n] + [0] * max(0, n - len(pred_bin))

        # Build predicted start/end row indices from boundaries.
        pred_starts = [i for i, b in enumerate(pred_bin) if b == 1 or i == 0]
        pred_ends: List[int] = []
        for i, s in enumerate(pred_starts):
            next_start = pred_starts[i + 1] if i + 1 < len(pred_starts) else n
            pred_ends.append(next_start - 1)

        pred_start_set = set(pred_starts)
        pred_end_set = set(pred_ends)

        human_start_rows = {
            i for i, v in enumerate(sub[START_LBL_COL]) if str(v).upper() == "TRUE"
        }
        human_end_rows = {
            i for i, v in enumerate(sub[END_LBL_COL]) if str(v).upper() == "TRUE"
        }

        human_seg = _human_story_membership(sub)
        # Embedding baseline considers every row to be inside some segment.
        llm_seg = [True] * n

        file_start = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        file_end = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        file_seg = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        file_inter = 0
        file_union = 0

        for i in range(n):
            h_s = i in human_start_rows
            p_s = i in pred_start_set
            if h_s and p_s: file_start["tp"] += 1
            elif p_s and not h_s: file_start["fp"] += 1
            elif h_s and not p_s: file_start["fn"] += 1
            else: file_start["tn"] += 1

            h_e = i in human_end_rows
            p_e = i in pred_end_set
            if h_e and p_e: file_end["tp"] += 1
            elif p_e and not h_e: file_end["fp"] += 1
            elif h_e and not p_e: file_end["fn"] += 1
            else: file_end["tn"] += 1

            h_g = human_seg[i]
            p_g = llm_seg[i]
            if h_g and p_g: file_seg["tp"] += 1
            elif p_g and not h_g: file_seg["fp"] += 1
            elif h_g and not p_g: file_seg["fn"] += 1
            else: file_seg["tn"] += 1

            if h_g and p_g: file_inter += 1
            if h_g or p_g: file_union += 1

        per_file[meeting_id] = {
            "start": _metrics_from_counts(file_start),
            "end": _metrics_from_counts(file_end),
            "segment": _metrics_from_counts(file_seg, inter=file_inter, union=file_union),
        }

        for k in start_counts: start_counts[k] += file_start[k]
        for k in end_counts: end_counts[k] += file_end[k]
        for k in seg_counts: seg_counts[k] += file_seg[k]
        inter += file_inter
        union += file_union

    overall = {
        "start": _metrics_from_counts(start_counts),
        "end": _metrics_from_counts(end_counts),
        "segment": _metrics_from_counts(seg_counts, inter=inter, union=union),
    }

    return {"per_file": per_file, "overall": overall}


def _metrics_from_counts(c: Dict[str, int], inter: int | None = None, union: int | None = None):
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    out = {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
           "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    if inter is not None and union is not None:
        out["iou"] = inter / union if union else 0.0
    return out


# --- CSV output in the project's 4-column format -------------------------

def write_labeled_csvs(
    df: pd.DataFrame,
    predicted_binary: Dict[str, List[int]],
    out_dir: str,
) -> List[str]:
    """Write one ``{meeting_id}_segmented.csv`` per transcript to ``out_dir``.

    The output CSV format matches the project's pipeline output:

        in/out/ambiguous, start, end, Transcript

    with ``start`` / ``end`` set by this baseline.  The ``in/out``
    column is copied through unchanged so the file remains usable for
    downstream inspection.
    """
    os.makedirs(out_dir, exist_ok=True)
    written: List[str] = []

    for meeting_id in sorted(set(df[MEETING_ID_COL])):
        sub = df[df[MEETING_ID_COL] == meeting_id].sort_values(START_COL).reset_index(drop=True)
        n = len(sub)
        pred_bin = predicted_binary.get(meeting_id, [0] * n)
        if len(pred_bin) != n:
            pred_bin = pred_bin[:n] + [0] * max(0, n - len(pred_bin))

        # Determine start rows (row 0 plus each boundary) and matching end rows.
        start_rows = sorted({0} | {i for i, b in enumerate(pred_bin) if b == 1})
        end_rows = set()
        for i, s in enumerate(start_rows):
            next_s = start_rows[i + 1] if i + 1 < len(start_rows) else n
            end_rows.add(next_s - 1)
        start_set = set(start_rows)

        out_path = os.path.join(out_dir, f"{meeting_id}_segmented.csv")
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["in/out/ambiguous", "start", "end", "Transcript"])
            for i, row in sub.iterrows():
                transcript = row[CAPTION_COL]
                if row[SPEAKER_COL]:
                    transcript = f"{row[SPEAKER_COL]}: {transcript}"
                writer.writerow([
                    row[IN_OUT_COL],
                    "TRUE" if i in start_set else "FALSE",
                    "TRUE" if i in end_rows else "FALSE",
                    transcript,
                ])
        written.append(out_path)

    return written


# --- Top-level convenience wrapper --------------------------------------

def evaluate(
    df: pd.DataFrame,
    predicted_boundaries: Dict[str, List[int]],
) -> Tuple[Dict[str, List[int]], Dict, Dict]:
    """Run both metric families against human ground truth.

    Returns:
        predicted_binary: per-meeting 0/1 boundary vectors
        pk_metrics: output of ``compute_pk_windiff``
        pipeline_metrics: output of ``compute_pipeline_metrics``
    """
    predicted_binary: Dict[str, List[int]] = {}
    for meeting_id in sorted(set(df[MEETING_ID_COL])):
        n = int((df[MEETING_ID_COL] == meeting_id).sum())
        predicted_binary[meeting_id] = boundaries_to_binary(
            predicted_boundaries.get(meeting_id, []), n
        )

    reference = human_boundary_labels(df)
    pk_m = compute_pk_windiff(predicted_binary, reference)
    pipe_m = compute_pipeline_metrics(df, predicted_binary)
    return predicted_binary, pk_m, pipe_m
