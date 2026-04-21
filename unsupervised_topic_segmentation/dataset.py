"""CSV loader for the Story Switch Detecting project.

Reads the project's 4-column CSV format:

    in/out/ambiguous, start, end, Transcript

and produces a pandas DataFrame in the schema expected by the rest of
the pipeline (core.py, baselines.py, eval.py):

    meeting_id, st, en, caption, speaker, in_out, start_lbl, end_lbl

Because our transcripts don't have real audio timestamps, ``st`` and
``en`` are set to the integer row index within the transcript.

Columns:
    meeting_id : transcript identifier (filename stem)
    st, en     : integer row index (``en = st + 1``)
    caption    : utterance text (with speaker prefix stripped if present)
    speaker    : speaker code parsed from the transcript prefix
                 (e.g. "PHIL: hi there" -> "PHIL"); empty if absent.
    in_out     : the original 'in/out/ambiguous' value (copied through)
    start_lbl  : 'TRUE' / 'FALSE' — human-annotated start marker
    end_lbl    : 'TRUE' / 'FALSE' — human-annotated end marker
"""

from __future__ import annotations

import os
import re
from glob import glob
from typing import Iterable, List, Tuple

import pandas as pd


# --- Constants used throughout the pipeline -----------------------------

MEETING_ID_COL = "meeting_id"
START_COL = "st"
END_COL = "en"
CAPTION_COL = "caption"
SPEAKER_COL = "speaker"
IN_OUT_COL = "in_out"
START_LBL_COL = "start_lbl"
END_LBL_COL = "end_lbl"


_SPEAKER_PREFIX_RE = re.compile(r"^\s*([A-Z][A-Z0-9_]{0,8})\s*:\s*(.*)$")


def _parse_speaker(line: str) -> Tuple[str, str]:
    """Split ``'NAME: utterance'`` into ``('NAME', 'utterance')``.

    Returns ``('', line)`` if no speaker prefix is detected.
    """
    if not isinstance(line, str):
        return "", ""
    m = _SPEAKER_PREFIX_RE.match(line)
    if m:
        return m.group(1), m.group(2).strip()
    return "", line.strip()


def load_csv(path: str, meeting_id: str | None = None) -> pd.DataFrame:
    """Load a single project CSV into the shared DataFrame schema.

    Args:
        path: Path to a 4-column CSV as described in the project's README.
        meeting_id: Override for the meeting_id column; defaults to the
            filename stem (no extension, trailing ``_labeled`` stripped).

    Returns:
        A DataFrame with one row per utterance in the pipeline's
        canonical schema.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)

    expected_cols = 4
    if df.shape[1] < expected_cols:
        raise ValueError(
            f"{path}: expected {expected_cols} columns, found {df.shape[1]}"
        )

    # Use positional access in case headers have minor typos.
    in_out = df.iloc[:, 0].astype(str)
    start_lbl = df.iloc[:, 1].astype(str)
    end_lbl = df.iloc[:, 2].astype(str)
    transcript = df.iloc[:, 3].astype(str)

    if meeting_id is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        meeting_id = stem[: -len("_labeled")] if stem.endswith("_labeled") else stem

    speakers, captions = zip(*[_parse_speaker(t) for t in transcript]) if len(transcript) else ([], [])

    out = pd.DataFrame({
        MEETING_ID_COL: [meeting_id] * len(df),
        START_COL: list(range(len(df))),
        END_COL: list(range(1, len(df) + 1)),
        CAPTION_COL: list(captions),
        SPEAKER_COL: list(speakers),
        IN_OUT_COL: in_out.tolist(),
        START_LBL_COL: start_lbl.tolist(),
        END_LBL_COL: end_lbl.tolist(),
    })
    return out


def load_directory(dir_path: str, pattern: str = "*.csv") -> pd.DataFrame:
    """Load every CSV in ``dir_path`` and concatenate into one DataFrame.

    Args:
        dir_path: Directory containing project CSVs.
        pattern: Glob pattern applied inside ``dir_path``.

    Returns:
        Concatenated DataFrame with one ``meeting_id`` per source file.
    """
    files = sorted(glob(os.path.join(dir_path, pattern)))
    if not files:
        raise FileNotFoundError(f"No CSVs matched {os.path.join(dir_path, pattern)}")
    frames = [load_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def load_files(paths: Iterable[str]) -> pd.DataFrame:
    """Load an explicit list of CSV paths and concatenate them."""
    frames: List[pd.DataFrame] = [load_csv(p) for p in paths]
    if not frames:
        raise ValueError("No input files provided.")
    return pd.concat(frames, ignore_index=True)


# --- Ground-truth extraction --------------------------------------------

def human_boundary_labels(df: pd.DataFrame) -> dict:
    """Convert the human start-markers into a per-meeting binary vector.

    For each meeting, returns a list of 0/1s (one per utterance) where
    ``1`` marks the first utterance of a new story (i.e. where
    ``start_lbl == 'TRUE'``).  The very first row is conventionally
    not counted as a boundary (there's no gap before it).

    Returns:
        Dict ``{meeting_id: [0, 0, 1, 0, 1, ...]}``.
    """
    labels: dict = {}
    for meeting_id in sorted(set(df[MEETING_ID_COL])):
        sub = df[df[MEETING_ID_COL] == meeting_id].sort_values(START_COL)
        start_flags = (sub[START_LBL_COL].str.upper() == "TRUE").astype(int).tolist()
        # Zero out the very first row — not a boundary in Pk/WinDiff sense.
        if start_flags:
            start_flags[0] = 0
        labels[meeting_id] = start_flags
    return labels
