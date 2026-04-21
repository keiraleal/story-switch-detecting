"""Naive baselines from Solbiati et al. (2021).

These are cheap sanity-check baselines used to contextualise the BERT/S-BERT
results:

- ``topic_segmentation_random``: place boundaries independently at random.
- ``topic_segmentation_even``:   place a boundary every N-th utterance.

Each function returns a dict mapping ``meeting_id`` to a list of
**gap indices** (boundary positions), matching the contract used by
``core.topic_segmentation``.
"""

from __future__ import annotations

from random import random
from typing import Dict, List

import pandas as pd


def topic_segmentation_random(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    random_threshold: float = 0.95,
) -> Dict[str, List[int]]:
    """Random baseline: a boundary at each utterance with prob 1 - threshold."""
    segments: Dict[str, List[int]] = {}
    for meeting_id in sorted(set(df[meeting_id_col_name])):
        n = int((df[meeting_id_col_name] == meeting_id).sum())
        segments[meeting_id] = [i for i in range(1, n) if random() > random_threshold]
    return segments


def topic_segmentation_even(
    df: pd.DataFrame,
    meeting_id_col_name: str,
    start_col_name: str,
    end_col_name: str,
    caption_col_name: str,
    every_n: int = 30,
) -> Dict[str, List[int]]:
    """Even baseline: place a boundary every ``every_n`` utterances."""
    segments: Dict[str, List[int]] = {}
    for meeting_id in sorted(set(df[meeting_id_col_name])):
        n = int((df[meeting_id_col_name] == meeting_id).sum())
        segments[meeting_id] = list(range(every_n, n, every_n))
    return segments
